mod model;

use anyhow::{anyhow, bail};
use candle_core::{CudaDevice, Device, Tensor, backend::BackendDevice, pickle};
use cpal::{
    FromSample, SampleFormat, SizedSample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use fundsp::hacker::*;
use hf_hub::{Cache, api::tokio::Api};
use opencv::{
    core::{Mat, MatTraitConst},
    highgui::{self, WND_PROP_VISIBLE},
    videoio::{
        CAP_ANY, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, VideoCapture,
        VideoCaptureTrait, VideoCaptureTraitConst,
    },
};
use std::io::stdin;

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

    let model = yolo11().await?;
    let device = Device::Cuda(CudaDevice::new(0)?);
    let mut camera = VideoCapture::new(0, CAP_ANY)?;
    if !camera.is_opened()? {
        bail!("failed to open camera");
    }
    camera.set(CAP_PROP_FRAME_WIDTH, WIDTH as _)?;
    camera.set(CAP_PROP_FRAME_HEIGHT, HEIGHT as _)?;
    camera.set(CAP_PROP_FPS, FPS as _)?;
    highgui::named_window(WINDOW, highgui::WINDOW_NORMAL)?;
    highgui::resize_window(WINDOW, WIDTH as _, HEIGHT as _)?;

    let mut raw = Mat::default();
    // let mut rgb = Mat::default();
    while next(&mut camera, &mut raw)? {
        // imgproc::cvt_color(src, dst, code, dst_cn)
        // let results = face.detect(&raw)?;
        // if detector.process(&frame, &mut mesh) {
        highgui::imshow(WINDOW, &raw)?;
        // }
    }

    highgui::destroy_window(WINDOW)?;
    Ok(())
}

async fn yolo11() -> anyhow::Result<Tensor> {
    let path = model::load("Ultralytics/YOLO11", "yolo11x-pose.pt").await?;
    let mut pairs = dbg!(pickle::read_all(path)?);
    Ok(pairs.pop().ok_or(anyhow!("no model found"))?.1)
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
