use rand::Rng;

struct LinearModel {
    poids: Vec<f64>,
}

impl LinearModel {
    fn new() -> Self {
        LinearModel { poids : vec![0.0;3] }
    }

	fn initPoids(&mut self) {
		for j in 0..self.poids.len() {
			self.poids[j] = rand::thread_rng().gen_range(-1.0..1.0);
		}
	}

    fn predict(&self, x1: f64, x2: f64) -> f64 {
        self.poids[0] + self.poids[1] * x1 + self.poids[2] * x2
    }

    // Fonction d'entraînement pour le modèle linéaire
    fn train(&mut self, all_inputs: Vec<Vec<f64>>, all_expected_outputs: Vec<f64>, alpha: f64, max_iter: usize, nb_errors : usize) -> Vec<f64> {
		let err_mod = (max_iter/nb_errors) as usize;
		let mut errors: Vec<f64> = Vec::new();
		for iter in 0..max_iter {
			if(iter % err_mod == 0) {
				errors.push(self.calculate_error(&all_inputs, &all_expected_outputs));
			}
            let k = rand::thread_rng().gen_range(0..all_inputs.len());
            let inputs_k = &all_inputs[k];
            let Yk = &all_expected_outputs[k];
			
			let mut Xk = vec![1.0;3];
			Xk[1] = inputs_k[0];
			Xk[2] = inputs_k[1];

			let yPredict = self.predict(Xk[1], Xk[2]);

			//let gXk = if yPredict >= 0.0 { 1.0 } else { -1.0 };

			// Mettez à jour les poids W
			self.poids[0] += alpha * (Yk - yPredict);
			self.poids[1] += alpha * (Yk - yPredict) * Xk[1];
			self.poids[2] += alpha * (Yk - yPredict) * Xk[2];
		}
		errors
    }
	
	fn calculate_error(&mut self, all_inputs: &Vec<Vec<f64>>, all_expected_outputs: &Vec<f64>) -> f64{
		let mut total_loss = 0.0;
		for iter in 0..all_inputs.len() {
			let result = self.predict(all_inputs[iter][0],all_inputs[iter][1]);
			total_loss += (all_expected_outputs[iter] - result).abs();
		}
		total_loss/(all_inputs.len() as f64)
	}
}

#[no_mangle]
pub extern "C" fn create_lm(nb_entree: i32, nb_couche: i32, nb_sortie: i32) -> *mut LinearModel {
	let mut LinearModel = Box::new(LinearModel::new());
	LinearModel.initPoids();
    Box::leak(LinearModel)
}

#[no_mangle]
pub extern "C" fn predict_lm(ptr: *mut LinearModel, input: *const f64, length: i32) -> f64 {
    let model = unsafe { &mut *ptr };
    let slice = unsafe { std::slice::from_raw_parts(input, length as usize) };
    let result = model.predict(slice[0], slice[1]);
	result
}

#[no_mangle]
pub extern "C" fn train_lm(ptr: *mut LinearModel, alpha : f64, max_iter : i32,  nb_errors : i32, input : *const f64, rowsIn: i32, colsIn: i32, output: *const f64, rowsOut: i32) -> *mut f64 {
    let model = unsafe { &mut *ptr };
    let sliceIn = unsafe { std::slice::from_raw_parts(input, (rowsIn * colsIn) as usize) };
    let mut input_data = vec![vec![0.0; colsIn as usize]; rowsIn as usize];
    for i in 0..rowsIn {
        let start = (i * colsIn) as usize;
        let end = ((i + 1) * colsIn) as usize;
        let row = &sliceIn[start..end];
        input_data[i as usize] = row.to_vec();
    }

	let output_data = unsafe { std::slice::from_raw_parts(output, rowsOut as usize) };
    (model.train(input_data, output_data.to_vec(), alpha, max_iter as usize, nb_errors as usize)).leak().as_mut_ptr()
}

pub struct MLP {
    n_entree: usize,
    n_sortie: usize,
    n_couche: usize,
    poids: Vec<Vec<Vec<f64>>>,
    entrees: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
}

impl MLP {
	fn new(mut nb_entree: usize, mut nb_couche: usize, nb_sortie: usize) -> Self {
		nb_entree+=1;
		nb_couche+=1;
        Self {
			n_entree: nb_entree,
			n_sortie: nb_sortie,
			n_couche: nb_couche,
			poids: vec![vec![vec![0.0;nb_entree]; nb_entree]; nb_couche],
			entrees: vec![vec![1.0;nb_entree]; nb_couche],
			deltas: vec![vec![0.0;nb_entree]; nb_couche],
        }
    }
	
	fn initPoids(&mut self) {
		for l in 0..self.n_couche {
			for i in 0..self.n_entree {
				for j in 0..self.n_entree {
					self.poids[l][i][j] = rand::thread_rng().gen_range(-1.0..1.0);
				}
			}
		}
	}

    fn propagate(&mut self, inputs: Vec<f64>, is_classification: bool) {
        //assert_eq!(inputs.len(), self.n_entree-1, "La taille des entrées ne correspond pas au nombre d'entrées du modèle");

        for j in 0..inputs.len() {
            self.entrees[0][j] = inputs[j];
        }

        for l in 1..self.n_couche {
            for j in 0..self.n_entree-1 {
                let mut sum = 0.0;

                for i in 0..self.n_entree {
                    sum += self.poids[l][i][j] * self.entrees[l - 1][i];
                }

                if l < self.n_couche - 1 || is_classification {
                    sum = sum.tanh();
                }

                self.entrees[l][j] = sum;
            }
        }
    }

    pub fn predict(&mut self, inputs: Vec<f64>, is_classification: bool) -> Vec<f64> {
        self.propagate(inputs, is_classification);
        self.entrees[self.n_couche-1].clone()
    }

    pub fn train(&mut self, all_inputs: Vec<Vec<f64>>, all_expected_outputs: Vec<Vec<f64>>, is_classification: bool, alpha: f64, max_iter: usize, nb_errors : usize) -> Vec<f64> {
		let mut errs: Vec<f64> = Vec::new();
		let err_mod = (max_iter/nb_errors) as usize;
		for iter in 0..max_iter {	
			if(iter % err_mod == 0){
				errs.push(self.calculate_loss(&all_inputs, &all_expected_outputs));
			}
			
            let k = rand::thread_rng().gen_range(0..all_inputs.len());
            let inputs_k = &all_inputs[k];
            let expected_outputs = &all_expected_outputs[k];

            self.propagate(inputs_k.to_vec(), is_classification);
			/*
			if((self.entrees[self.n_couche-1][0] - expected_outputs[0]).abs() < 0.5){
				continue;
			}
			*/
            for j in 0..self.n_sortie {
                self.deltas[self.n_couche - 1][j] = self.entrees[self.n_couche - 1][j] - expected_outputs[j];

                if is_classification {
                    self.deltas[self.n_couche - 1][j] *= 1.0 - self.entrees[self.n_couche - 1][j].powi(2);
                }
            }

            for l in (1..self.n_couche).rev() {
                for i in 0..self.n_entree {
                    let mut sum = 0.0;

                    for j in 0..self.n_entree {
                        sum += self.poids[l][i][j] * self.deltas[l][j];
                    }

                    self.deltas[l - 1][i] = (1.0 - self.entrees[l - 1][i].powi(2)) * sum;
                }
            }

            for l in 1..self.n_couche {
                for i in 0..self.n_entree {
                    for j in 0..self.n_entree {
                        self.poids[l][i][j] -= alpha * self.entrees[l - 1][i] * self.deltas[l][j];
                    }
                }
            }
			/*
			if(iter == max_iter-1){
				for l in 0..self.n_couche {
					for j in 0..self.n_entree {
						println!("{}, {}, {}", l, j, self.deltas[l][j]);
					}
				}
			}*/
        }

		errs
    }
	
	fn calculate_loss(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) -> f64 {
		let mut total_loss = 0.0;
		for (input, target) in inputs.iter().zip(targets.iter()) {
			let prediction = self.predict(input.to_vec(), true);
			//println!("{}", prediction[0]);
			let errors: Vec<f64> = prediction.iter().zip(target.iter()).map(|(p, t)| t - p).collect();
			total_loss += errors.iter().map(|e| e.powi(2)).sum::<f64>();
		}

		total_loss / (inputs.len() as f64)
	}
}

#[no_mangle]
pub extern "C" fn create_mlp(nb_entree: i32, nb_couche: i32, nb_sortie: i32) -> *mut MLP {
	let mut mlp = Box::new(MLP::new(nb_entree as usize, nb_couche as usize, nb_sortie as usize));
	mlp.initPoids();
    Box::leak(mlp)
}

#[no_mangle]
pub extern "C" fn predict_mlp(ptr: *mut MLP, input: *const f64, length: i32) -> *mut f64 {
    let model = unsafe { &mut *ptr };
    let slice = unsafe { std::slice::from_raw_parts(input, length as usize) };
    let result = model.predict(slice.to_vec(), true);
    result.leak().as_mut_ptr()
}

#[no_mangle]
pub extern "C" fn train_mlp(ptr: *mut MLP, alpha : f64, max_iter : i32, nb_errors : i32, input : *const f64, rowsIn: i32, colsIn: i32, output: *const f64, rowsOut: i32, colsOut: i32) -> *mut f64 {
    let model = unsafe { &mut *ptr };
    let sliceIn = unsafe { std::slice::from_raw_parts(input, (rowsIn * colsIn) as usize) };
    let mut input_data = vec![vec![0.0; colsIn as usize]; rowsIn as usize];
    for i in 0..rowsIn {
        let start = (i * colsIn) as usize;
        let end = ((i + 1) * colsIn) as usize;
        let row = &sliceIn[start..end];
        input_data[i as usize] = row.to_vec();
		//println!("{}, {}",row[0], row[1]);
    }

	let sliceOut = unsafe { std::slice::from_raw_parts(output, (rowsOut * colsOut) as usize) };
    let mut output_data =  vec![vec![0.0; colsOut as usize]; rowsOut as usize];
    for i in 0..rowsOut {
        let start = (i * colsOut) as usize;
        let end = ((i + 1) * colsOut) as usize;
        let row = &sliceOut[start..end];
        output_data[i as usize] = row.to_vec();
    }

    model.train(input_data, output_data, true, alpha, max_iter as usize, nb_errors as usize).leak().as_mut_ptr()
}