#![recursion_limit = "256"]

use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::nn::{self, loss::CrossEntropyLossConfig, DropoutConfig};
use burn::optim::Optimizer;
use burn::optim::{AdamConfig, GradientsParams};
use burn::record::{CompactRecorder, Recorder};
use burn::tensor::backend::Backend;
use burn::tensor::{Int, Tensor};
use burn_autodiff::Autodiff;
use burn_ndarray::NdArray;
use polars::datatypes::DataType::Float64;
use polars::prelude::{CsvReadOptions, DataFrameJoinOps, JoinType, SerReader};
use rand::rng;
use rand::seq::SliceRandom;
use std::error::Error;
use std::fs;

#[derive(Module, Debug)]
struct MyModel<B: Backend> {
    layer1: nn::Linear<B>,
    layer2: nn::Linear<B>,
    layer3: nn::Linear<B>,
}

impl<B: Backend> MyModel<B> {
    fn new(device: &B::Device, d_in: usize, d_hidden: usize, d_out: usize) -> Self {
        Self {
            layer1: nn::LinearConfig::new(d_in, d_hidden).init(device),
            layer2: nn::LinearConfig::new(d_hidden, d_hidden).init(device),
            layer3: nn::LinearConfig::new(d_hidden, d_out).init(device),
        }
    }

    fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.layer1.forward(x);
        let x = nn::Relu::new().forward(x);
        let x = self.layer2.forward(x);
        let x = nn::Relu::new().forward(x);
        let x = DropoutConfig::new(0.3).init().forward(x);
        self.layer3.forward(x)
    }

    fn save(&self, path: &str) -> Result<(), Box<dyn Error>> {
        let recorder = CompactRecorder::new();
        recorder
            .record(self.clone().into_record(), path.into())
            .map_err(|e| format!("Erreur sauvegarde: {}", e))?;
        Ok(())
    }

    fn load(
        path: &str,
        device: &B::Device,
        d_in: usize,
        d_hidden: usize,
        d_out: usize,
    ) -> Result<Self, Box<dyn Error>> {
        let recorder = CompactRecorder::new();
        let record = recorder
            .load(path.into(), device)
            .map_err(|e| format!("Erreur chargement: {}", e))?;

        let mut model = Self::new(device, d_in, d_hidden, d_out);
        model = model.load_record(record);
        Ok(model)
    }
}

fn load_data_extended() -> Result<(Vec<f32>, Vec<i64>, usize), Box<dyn Error>> {
    let anime_df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("anime.csv".into()))?
        .finish()?;

    let rating_df = CsvReadOptions::default()
        .with_has_header(true)
        .try_into_reader_with_file_path(Some("rating.csv".into()))?
        .finish()?;

    let joined = rating_df.join(
        &anime_df,
        ["anime_id"],
        ["anime_id"],
        JoinType::Inner.into(),
        None,
    )?;

    let valid_mask = joined.column("rating")?.is_not_null()
        & joined.column("genre")?.is_not_null()
        & joined.column("episodes")?.is_not_null();
    let clean_data = joined.filter(&valid_mask)?;

    let labels: Vec<i64> = clean_data
        .column("rating")?
        .i64()?
        .into_iter()
        .map(|opt| if opt.unwrap_or(0) >= 7 { 1 } else { 0 })
        .collect();

    let genres_series = clean_data.column("genre")?;
    let unique_genres: Vec<String> = genres_series
        .str()?
        .into_iter()
        .flat_map(|opt| opt.unwrap_or("").split(", "))
        .map(|s| s.to_string())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let episodes: Vec<f32> = clean_data
        .column("episodes")?
        .str()?
        .into_iter()
        .map(|opt| {
            opt.and_then(|s| s.trim().parse::<i32>().ok())
                .filter(|&x| x > 0)
                .map(|x| (x as f32).ln())
                .unwrap_or(0.0)
        })
        .collect();

    let rating_counts: Vec<f32> = clean_data
        .column("members")?
        .i64()?
        .into_iter()
        .map(|opt| (opt.unwrap_or(0) as f32).ln())
        .collect();

    let avg_ratings: Vec<f32> = clean_data
        .column("rating")?
        .cast(&Float64)?
        .f64()?
        .into_iter()
        .map(|opt| match opt {
            Some(val) if val >= 0.0 => (val as f32) / 10.0,
            _ => 0.5,
        })
        .collect();

    let nb_genre_features = unique_genres.len();
    let nb_total_features = nb_genre_features + 3;

    let mut features: Vec<f32> = Vec::new();

    for (i, genre_str) in genres_series.str()?.into_iter().enumerate() {
        let mut row = vec![0.0; nb_genre_features];
        if let Some(glist) = genre_str {
            for g in glist.split(", ") {
                if let Some(pos) = unique_genres.iter().position(|x| x == g) {
                    row[pos] = 1.0;
                }
            }
        }

        row.push(episodes[i]);
        row.push(rating_counts[i]);
        row.push(avg_ratings[i]);

        features.extend(row);
    }

    let max_eps = episodes.iter().cloned().fold(f32::MIN, f32::max);
    let max_count = rating_counts.iter().cloned().fold(f32::MIN, f32::max);
    let min_rating = avg_ratings.iter().cloned().fold(f32::MAX, f32::min);
    let max_rating = avg_ratings.iter().cloned().fold(f32::MIN, f32::max);

    let avg_ratings: Vec<f32> = avg_ratings
        .into_iter()
        .map(|x| (x - min_rating) / (max_rating - min_rating))
        .collect();

    for chunk in features.chunks_mut(nb_genre_features + 3) {
        chunk[nb_genre_features + 0] /= max_eps;
        chunk[nb_genre_features + 1] /= max_count;
    }

    println!(
        "Features étendues: {} genres + 3 autres = {} features totales",
        nb_genre_features, nb_total_features
    );

    println!("labels: {}", labels.len());
    println!("episodes: {}", episodes.len());
    println!("rating_counts: {}", rating_counts.len());
    println!("avg_ratings: {}", avg_ratings.len());
    println!("genres_series: {}", genres_series.len());

    Ok((features, labels, nb_total_features))
}

fn model_exists(path: &str) -> bool {
    fs::metadata(path).is_ok()
}

fn main() -> Result<(), Box<dyn Error>> {
    type MyBackend = Autodiff<NdArray>;
    let device = <MyBackend as Backend>::Device::default();

    let model_path = "anime_model.mpk";

    let (features, labels, nb_features) = load_data_extended()?;
    let n_samples = labels.len();

    let mut indices: Vec<usize> = (0..n_samples).collect();
    indices.shuffle(&mut rng());

    let split_idx = (0.8 * n_samples as f32).round() as usize;
    let (train_idx, test_idx) = indices.split_at(split_idx);

    let train_features: Vec<f32> = train_idx
        .iter()
        .flat_map(|&i| {
            let start = i * nb_features;
            let end = start + nb_features;
            features[start..end].to_vec()
        })
        .collect();

    let test_features: Vec<f32> = test_idx
        .iter()
        .flat_map(|&i| {
            let start = i * nb_features;
            let end = start + nb_features;
            features[start..end].to_vec()
        })
        .collect();

    let train_labels: Vec<i64> = train_idx.iter().map(|&i| labels[i]).collect();
    let test_labels: Vec<i64> = test_idx.iter().map(|&i| labels[i]).collect();

    assert_eq!(train_features.len(), train_labels.len() * nb_features);
    assert_eq!(test_features.len(), test_labels.len() * nb_features);

    let x_train = Tensor::<MyBackend, 1>::from_floats(&*train_features, &device)
        .reshape([train_labels.len(), nb_features]);
    let y_train = Tensor::<MyBackend, 1, Int>::from_ints(&*train_labels, &device)
        .reshape([train_labels.len()]);

    let x_test = Tensor::<MyBackend, 1>::from_floats(&*test_features, &device)
        .reshape([test_labels.len(), nb_features]);
    let y_test =
        Tensor::<MyBackend, 1, Int>::from_ints(&*test_labels, &device).reshape([test_labels.len()]);

    println!(
        "Dataset étendu: {} échantillons avec {} features",
        n_samples, nb_features
    );
    println!(
        "Train: {} échantillons, Test: {} échantillons",
        train_labels.len(),
        test_labels.len()
    );

    let mut model = if model_exists(model_path) {
        println!("Chargement du modèle existant...");
        MyModel::<MyBackend>::load(model_path, &device, nb_features, 32, 2)?
    } else {
        println!("Création d'un nouveau modèle...");
        MyModel::<MyBackend>::new(&device, nb_features, 16, 2)
    };

    let loss_fn = CrossEntropyLossConfig::new().init(&device);
    let optimizer_config =
        AdamConfig::new().with_grad_clipping(Option::from(GradientClippingConfig::Norm(1.0)));
    let mut optimizer = optimizer_config.init();

    let mut best_acc = 0.0;
    let mut patience_counter = 0;
    let patience = 100;
    let lr = 0.001;
    let batch_size = 1024;

    for epoch in 0..1000 {
        for batch_start in (0..train_labels.len()).step_by(batch_size) {
            let batch_end = (batch_start + batch_size).min(train_labels.len());

            let x_batch = x_train
                .clone()
                .slice([batch_start..batch_end, 0..nb_features]);
            let y_batch = y_train.clone().slice([batch_start..batch_end]);

            let logits = model.forward(x_batch);
            let loss = loss_fn.forward(logits.clone(), y_batch);

            let grads = GradientsParams::from_grads(loss.backward(), &model);
            model = optimizer.step(lr, model, grads);
        }

        if epoch % 10 == 0 {
            let logits_train = model.forward(x_train.clone());
            let loss_train = loss_fn.forward(logits_train.clone(), y_train.clone());

            let preds_test = model.forward(x_test.clone()).argmax(1);
            let preds_vec: Vec<i64> = preds_test.into_data().to_vec().unwrap();
            let acc_test = preds_vec
                .iter()
                .zip(test_labels.iter())
                .filter(|(p, l)| p == l)
                .count() as f32
                / test_labels.len() as f32;

            println!(
                "Epoch {}: perte train = {:.4}, précision test = {:.2}%",
                epoch,
                loss_train.into_scalar(),
                acc_test * 100.0
            );

            if acc_test > best_acc {
                best_acc = acc_test;
                patience_counter = 0;
                println!("Sauvegarde du modèle...");
                model.save(model_path)?;
                println!("Modèle sauvegardé dans '{}'", model_path);
            } else {
                patience_counter += 10;
            }

            if patience_counter >= patience {
                println!(
                    "Early stopping à l'epoch {} (précision test ne s'améliore plus)",
                    epoch
                );
                break;
            }
        }
    }

    let preds = model.forward(x_test.clone()).argmax(1);
    let preds_vec: Vec<i64> = preds.into_data().to_vec().unwrap();
    let acc = preds_vec
        .iter()
        .zip(test_labels.iter())
        .filter(|(p, l)| p == l)
        .count() as f32
        / test_labels.len() as f32;

    println!("Précision finale sur le test set: {:.2}%", acc * 100.0);

    Ok(())
}
