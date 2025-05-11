mod model;

use anyhow::{anyhow, bail};
use cpal::{
    FromSample, SampleFormat, SizedSample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use fundsp::hacker::*;
use opencv::{
    core::{Mat, MatTraitConst, MatTraitConstManual},
    highgui::{self, WND_PROP_VISIBLE},
    videoio::{
        CAP_ANY, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, VideoCapture,
        VideoCaptureTrait, VideoCaptureTraitConst,
    },
};
use std::io::stdin;
use tch::{CModule, Device, Kind, Tensor};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // audio().await?;
    video().await?;
    Ok(())
}

async fn video() -> anyhow::Result<()> {
    fn next(camera: &mut VideoCapture, frame: &mut Mat) -> anyhow::Result<bool> {
        if camera.read(frame)? {
            if frame.size()?.empty()
                || highgui::poll_key()? == 24
                || highgui::get_window_property(WINDOW, WND_PROP_VISIBLE)? < 1.0
            {
                Ok(false)
            } else {
                Ok(true)
            }
        } else {
            Ok(false)
        }
    }

    const WINDOW: &str = "La Brousse À Gigante";
    const WIDTH: usize = 640;
    const HEIGHT: usize = 480;
    const FPS: usize = 30;

    // let (input, model) = yolo11().await?;
    // let mut inputs = HashMap::new();
    let device = Device::Cuda(0);
    let model = yolo11(device).await?;
    let mut camera = VideoCapture::new(0, CAP_ANY)?;
    if !camera.is_opened()? {
        bail!("failed to open camera");
    }
    camera.set(CAP_PROP_FRAME_WIDTH, WIDTH as _)?;
    camera.set(CAP_PROP_FRAME_HEIGHT, HEIGHT as _)?;
    camera.set(CAP_PROP_FPS, FPS as _)?;
    highgui::named_window(WINDOW, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(WINDOW, WIDTH as _, HEIGHT as _)?;

    let mut frame = Mat::default();
    while next(&mut camera, &mut frame)? {
        let input = to_tensor(&frame, device)?;
        println!("INPUT: {input:?}");
        let output = model.forward_ts(&[input])?.squeeze();
        println!("OUTPUT: {output:?}");
        highgui::imshow(WINDOW, &frame)?;
    }

    highgui::destroy_window(WINDOW)?;
    Ok(())
}

pub fn to_tensor(frame: &Mat, device: Device) -> anyhow::Result<Tensor> {
    let (height, width) = (frame.rows() as i64, frame.cols() as i64);
    let data = frame.data_bytes()?; // &[u8] in B, G, R, B, G, R, …
    Ok(
        Tensor::from_data_size(data, &[height, width, 3], Kind::Uint8)
            .to_device(device)
            .permute([2, 0, 1])
            .index_select(0, &Tensor::from_slice(&[2i64, 1, 0]).to_device(device))
            .to_kind(Kind::Float)
            .divide_scalar(255.0)
            .unsqueeze(0),
    )
}

async fn yolo11(device: Device) -> anyhow::Result<CModule> {
    let path = model::load("Ultralytics/YOLO11", "yolo11x-pose.pt").await?;
    let model = CModule::load_on_device(&path, device)?;
    println!("Model loaded at path '{path:?}' on {device:?}.");
    for (key, value) in model.named_parameters()? {
        println!("=> {key}: {value:?}");
    }
    Ok(model)
}

async fn audio() -> anyhow::Result<()> {
    let device = cpal::default_host()
        .default_output_device()
        .ok_or(anyhow!("no output device available"))?;
    let configuration = device.default_output_config()?;
    match configuration.sample_format() {
        SampleFormat::F32 => run::<f32>(device, configuration.into())?,
        SampleFormat::I16 => run::<i16>(device, configuration.into())?,
        SampleFormat::U16 => run::<u16>(device, configuration.into())?,
        SampleFormat::I8 => run::<i8>(device, configuration.into())?,
        SampleFormat::U8 => run::<u8>(device, configuration.into())?,
        SampleFormat::I32 => run::<i32>(device, configuration.into())?,
        SampleFormat::U32 => run::<u32>(device, configuration.into())?,
        SampleFormat::I64 => run::<i64>(device, configuration.into())?,
        SampleFormat::U64 => run::<u64>(device, configuration.into())?,
        SampleFormat::F64 => run::<f64>(device, configuration.into())?,
        _ => bail!("unsupported sample format"),
    }
    Ok(())
}

fn run<T: SizedSample + FromSample<f32>>(
    device: cpal::Device,
    configuration: cpal::StreamConfig,
) -> anyhow::Result<()> {
    let sample_rate = configuration.sample_rate.0 as f64;
    let channels = configuration.channels as usize;
    let mut instrument = instrument();
    instrument.set_sample_rate(sample_rate);
    instrument.allocate();
    let stream = device.build_output_stream(
        &configuration,
        move |data, _| write::<T, _>(data, channels, &mut instrument),
        |error| eprintln!("an error occurred on stream: {}", error),
        None,
    )?;
    stream.play()?;
    stdin().read_line(&mut String::new())?;
    Ok(())
}

fn instrument() -> impl AudioNode {
    let sound = 0.2 * (organ_hz(midi_hz(57.0)) + organ_hz(midi_hz(61.0)) + organ_hz(midi_hz(64.0)));
    let sound = sound >> pan(0.0);
    let sound = sound >> (chorus(0, 0.0, 0.01, 0.2) | chorus(1, 0.0, 0.01, 0.2));
    let sound =
        sound >> (declick() | declick()) >> (dcblock() | dcblock()) >> limiter_stereo(1.0, 5.0);
    sound.0
}

fn write<T, A: AudioNode>(output: &mut [T], channels: usize, instrument: &mut A)
where
    T: SizedSample + FromSample<f32>,
{
    for frame in output.chunks_mut(channels) {
        let sample = instrument.get_stereo();
        let left = T::from_sample(sample.0);
        let right = T::from_sample(sample.1);

        for (channel, sample) in frame.iter_mut().enumerate() {
            if channel & 1 == 0 {
                *sample = left;
            } else {
                *sample = right;
            }
        }
    }
}
