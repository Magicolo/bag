// use mediapipe_rs::tasks::vision::{FaceLandmarker, FaceLandmarkerBuilder};
// use std::path::Path;
// use tokio::{
//     fs::{self, File},
//     io::AsyncWriteExt,
// };

// pub trait Build {
//     async fn build(self) -> anyhow::Result<FaceLandmarker>;
// }

// impl Build for FaceLandmarkerBuilder {
//     async fn build(self) -> anyhow::Result<FaceLandmarker> {
//         const LOCAL: &str = "models/face_landmarker.task";
//         const REMOTE: &str =
//             "https://storage.googleapis.com/mediapipe-tasks/face_landmarker/face_landmarker.task";

//         let local = Path::new(LOCAL);
//         if !local.exists() {
//             println!("Downloading model from '{REMOTE}' into '{LOCAL}'...");
//             let response = reqwest::get(REMOTE).await?;
//             let body = response.bytes().await?;
//             File::create(local).await?.write_all(&body).await?;
//         }

//         let bytes = fs::read(local).await?;
//         Ok(self.build_from_buffer(bytes)?)
//     }
// }

use hf_hub::{Cache, api::tokio::Api};
use std::path::PathBuf;

pub async fn load(repository: &str, name: &str) -> anyhow::Result<PathBuf> {
    Ok(match Cache::from_env().model(repository.into()).get(name) {
        Some(path) => path,
        None => Api::new()?.model(repository.into()).download(name).await?,
    })
}
