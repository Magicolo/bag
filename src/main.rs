mod model;

use anyhow::{anyhow, bail};
use cpal::{
    FromSample, SampleFormat, SizedSample,
    traits::{DeviceTrait, HostTrait, StreamTrait},
};
use fundsp::hacker::*;
use mediapipe_rs::tasks::vision::FaceLandmarkerBuilder;
use model::Build;
use opencv::{
    core::{Mat, MatTraitConst},
    videoio::{
        CAP_ANY, CAP_PROP_FPS, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_WIDTH, VideoCapture,
        VideoCaptureTrait, VideoCaptureTraitConst,
    },
};
use std::io::stdin;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    video().await?;
    Ok(())
}

async fn video() -> anyhow::Result<()> {
    const WINDOW: &str = "La Brousse À Gigante";

    let mut camera = VideoCapture::new(0, CAP_ANY)?;
    if !camera.is_opened()? {
        bail!("failed to open camera");
    }
    camera.set(CAP_PROP_FRAME_WIDTH, 640.0)?;
    camera.set(CAP_PROP_FRAME_HEIGHT, 480.0)?;
    camera.set(CAP_PROP_FPS, 30.0)?;
    // highgui::named_window(WINDOW, highgui::WINDOW_AUTOSIZE)?;

    // let face = FaceLandmarkerBuilder::new()
    //     .num_faces(1)
    //     .min_face_detection_confidence(0.5)
    //     .min_face_presence_confidence(0.5)
    //     .min_tracking_confidence(0.5)
    //     .output_face_blendshapes(true)
    //     .build()
    //     .await?;
    // face.detect(input);
    // let mut mesh = FaceMesh::default();
    // let mut detector = FaceMeshDetector::default();
    let mut frame = Mat::default();
    while camera.read(&mut frame)? {
        if frame.size()?.empty() {
            println!("skip empty frame");
            continue;
        } else {
            println!("got frame {:?}!", frame.size())
        }

        // if detector.process(&frame, &mut mesh) {
        // highgui::imshow(WINDOW, &frame)?;
        // }
    }
    Ok(())
}

fn audio() -> anyhow::Result<()> {
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
