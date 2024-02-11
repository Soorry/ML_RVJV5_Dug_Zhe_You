use rand::Rng;
use nalgebra::DMatrix;

struct LinearModel {
    poids: Vec<f64>,
}

impl LinearModel {
    fn new(nb_entree : usize) -> Self {
        LinearModel { poids : vec![0.0;nb_entree+1] }
    }

	fn init_poids(&mut self) {
		for j in 0..self.poids.len() {
			self.poids[j] = rand::thread_rng().gen_range(-1.0..1.0);
		}
	}

    fn predict(&self, x : Vec<f64>) -> f64 {
		let mut output = self.poids[0];
		if x.len() > 1 //Classification
		{
			for i in 0..x.len() {
				output += self.poids[i+1] * x[i];
			}	
		}
		else { //Regression
			for i in 1..self.poids.len() {
				output += self.poids[i] * x[0].powf(i as f64);
			}	
		}
		output
	}

    fn train(&mut self, all_inputs: Vec<Vec<f64>>, all_expected_outputs: Vec<f64>, alpha: f64, max_iter: usize, nb_errors : usize) -> Vec<f64> {
		let err_mod = (max_iter/nb_errors) as usize;
		let mut errors: Vec<f64> = Vec::new();
		for iter in 0..max_iter {
			if iter % err_mod == 0 {
				errors.push(self.calculate_error(&all_inputs, &all_expected_outputs));
			}
            let k = rand::thread_rng().gen_range(0..all_inputs.len());
            let inputs_k = &all_inputs[k];
            let yk = &all_expected_outputs[k];

			let ypredict = self.predict(inputs_k.to_vec());

			self.poids[0] += alpha * (yk - ypredict);
			for i in 0..inputs_k.len() {
				self.poids[i+1] += alpha * (yk - ypredict) * inputs_k[i];
			}
		}
		errors
    }
	
	fn calculate_error(&mut self, all_inputs: &Vec<Vec<f64>>, all_expected_outputs: &Vec<f64>) -> f64{
		let mut total_loss = 0.0;
		for iter in 0..all_inputs.len() {
			let result = self.predict(all_inputs[iter].to_vec());
			total_loss += (all_expected_outputs[iter] - result).abs();
		}
		total_loss/(all_inputs.len() as f64)
	}
}

#[no_mangle]
extern "C" fn create_lm(nb_entree: i32) -> *mut LinearModel {
	let mut linear_model = Box::new(LinearModel::new(nb_entree as usize));
	linear_model.init_poids();
    Box::leak(linear_model)
}

#[no_mangle]
extern "C" fn predict_lm(ptr: *mut LinearModel, input: *const f64, length: i32) -> f64 {
    let model = unsafe { &mut *ptr };
    let slice = unsafe { std::slice::from_raw_parts(input, length as usize) };
    let result = model.predict(slice.to_vec());
	result
}

#[no_mangle]
extern "C" fn train_lm(ptr: *mut LinearModel, alpha : f64, max_iter : i32,  nb_errors : i32, input : *const f64, rows_in: i32, cols_in: i32, output: *const f64, rows_out: i32) -> *mut f64 {
    let model = unsafe { &mut *ptr };
    let slice_in = unsafe { std::slice::from_raw_parts(input, (rows_in * cols_in) as usize) };
    let mut input_data = vec![vec![0.0; cols_in as usize]; rows_in as usize];
    for i in 0..rows_in {
        let start = (i * cols_in) as usize;
        let end = ((i + 1) * cols_in) as usize;
        let row = &slice_in[start..end];
        input_data[i as usize] = row.to_vec();
    }

	let output_data = unsafe { std::slice::from_raw_parts(output, rows_out as usize) };
    (model.train(input_data, output_data.to_vec(), alpha, max_iter as usize, nb_errors as usize)).leak().as_mut_ptr()
}

struct MLP {
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
	
	fn init_poids(&mut self) {
		for l in 0..self.n_couche {
			for i in 0..self.n_entree {
				for j in 0..self.n_entree {
					self.poids[l][i][j] = rand::thread_rng().gen_range(-1.0..1.0);
				}
			}
		}
	}

    fn propagate(&mut self, inputs: Vec<f64>, is_classification: bool) {
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

    fn predict(&mut self, inputs: Vec<f64>, is_classification: bool) -> Vec<f64> {
        self.propagate(inputs, is_classification);
        self.entrees[self.n_couche-1].clone()
    }

    fn train(&mut self, all_inputs: Vec<Vec<f64>>, all_expected_outputs: Vec<Vec<f64>>, is_classification: bool, alpha: f64, max_iter: usize, nb_errors : usize) -> Vec<f64> {
		let mut errs: Vec<f64> = Vec::new();
		let err_mod = (max_iter/nb_errors) as usize; //On calcule le modulo pour obtenir le nombre d'erreurs souhaité
		for iter in 0..max_iter {	
			if iter % err_mod == 0 {
				errs.push(self.calculate_loss(&all_inputs, &all_expected_outputs));
			}
			
            let k = rand::thread_rng().gen_range(0..all_inputs.len());
            let inputs_k = &all_inputs[k];
            let expected_outputs = &all_expected_outputs[k];

            self.propagate(inputs_k.to_vec(), is_classification);

			if (self.entrees[self.n_couche-1][0] - expected_outputs[0]).abs() < 0.01 {
				continue;
			}
			
			//Calcul des deltas de la dernière couche
            for j in 0..self.n_sortie {
                self.deltas[self.n_couche - 1][j] = self.entrees[self.n_couche - 1][j] - expected_outputs[j];

                if is_classification {
                    self.deltas[self.n_couche - 1][j] *= 1.0 - self.entrees[self.n_couche - 1][j].powi(2);
                }
            }

			//Backpropagation
            for l in (1..self.n_couche).rev() {
                for i in 0..self.n_entree {
                    let mut sum = 0.0;

                    for j in 0..self.n_entree {
                        sum += self.poids[l][i][j] * self.deltas[l][j];
                    }

                    self.deltas[l - 1][i] = (1.0 - self.entrees[l - 1][i].powi(2)) * sum;
                }
            }

			//Mise à jour des poids
            for l in 1..self.n_couche {
                for i in 0..self.n_entree {
                    for j in 0..self.n_entree {
                        self.poids[l][i][j] -= alpha * self.entrees[l - 1][i] * self.deltas[l][j];
                    }
                }
			}
        }

		errs
    }
	
	fn calculate_loss(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) -> f64 {
		let mut total_loss = 0.0;
		for (input, target) in inputs.iter().zip(targets.iter()) {
			let prediction = self.predict(input.to_vec(), true);
			let errors: Vec<f64> = prediction.iter().zip(target.iter()).map(|(p, t)| t - p).collect();
			total_loss += errors.iter().map(|e| e.powi(2)).sum::<f64>();
		}

		total_loss / (inputs.len() as f64)
	}
}

#[no_mangle]
extern "C" fn create_mlp(nb_entree: i32, nb_couche: i32, nb_sortie: i32) -> *mut MLP {
	let mut mlp = Box::new(MLP::new(nb_entree as usize, nb_couche as usize, nb_sortie as usize));
	mlp.init_poids();
    Box::leak(mlp)
}

#[no_mangle]
extern "C" fn predict_mlp(ptr: *mut MLP, input: *const f64, length: i32) -> *mut f64 {
    let model = unsafe { &mut *ptr };
    let slice = unsafe { std::slice::from_raw_parts(input, length as usize) };
    let result = model.predict(slice.to_vec(), true);
    result.leak().as_mut_ptr()
}

#[no_mangle]
extern "C" fn train_mlp(ptr: *mut MLP, alpha : f64, max_iter : i32, nb_errors : i32, input : *const f64, rows_in: i32, cols_in: i32, output: *const f64, rows_out: i32, cols_out: i32) -> *mut f64 {
    let model = unsafe { &mut *ptr };
    let slice_in = unsafe { std::slice::from_raw_parts(input, (rows_in * cols_in) as usize) };
    let mut input_data = vec![vec![0.0; cols_in as usize]; rows_in as usize];
    for i in 0..rows_in {
        let start = (i * cols_in) as usize;
        let end = ((i + 1) * cols_in) as usize;
        let row = &slice_in[start..end];
        input_data[i as usize] = row.to_vec();
    }

	let slice_out = unsafe { std::slice::from_raw_parts(output, (rows_out * cols_out) as usize) };
    let mut output_data =  vec![vec![0.0; cols_out as usize]; rows_out as usize];
    for i in 0..rows_out {
        let start = (i * cols_out) as usize;
        let end = ((i + 1) * cols_out) as usize;
        let row = &slice_out[start..end];
        output_data[i as usize] = row.to_vec();
    }

    model.train(input_data, output_data, true, alpha, max_iter as usize, nb_errors as usize).leak().as_mut_ptr()
}

struct Rbf {
    poids: Vec<f64>,
	exemple_x : Vec<Vec<f64>>,
	exemple_y : Vec<f64>,
	gammma : f64
}

impl Rbf {
    fn new(nb_poids : usize, gam : f64, x : Vec<Vec<f64>>, y : Vec<f64>) -> Self {
        Rbf { poids : vec![0.0;nb_poids], exemple_x : x, exemple_y : y, gammma : gam }
    }

	fn init_poids(&mut self) {
		let n = self.exemple_x.len();
		let mut matrix = DMatrix::<f64>::zeros(n, n);
		//On rempli la matrice
		for i in 0..n {
			for j in 0..n {
				let diff_x1 = self.exemple_x[i][0] - self.exemple_x[j][0];
				let diff_x2 = self.exemple_x[i][1] - self.exemple_x[j][1];
				let mag_x = diff_x1*diff_x1 + diff_x2*diff_x2;
				matrix[(i,j)] = (-self.gammma * mag_x).exp();
			}
		}
		
		//On l'inverse
		let matrix_inv = matrix.try_inverse().unwrap();
				
		let mut matrix_inv_vec = vec![vec![0.0;n];n]; 
				
		for i in 0..n {
			for j in 0..n {
				matrix_inv_vec[i][j] = matrix_inv[(i,j)];
			}
		}
				
		let mut new_poids = Vec::with_capacity(self.exemple_y.len());
		
		//On multiplie la matrice et le vecteur des résultats attendus
		for row in matrix_inv_vec {
			let dot_product = row.iter().zip(self.exemple_y.iter()).map(|(a, b)| a * b).sum();
			new_poids.push(dot_product);
		}
		
		self.poids = new_poids;
	}

    fn predict(&self, x1: f64, x2: f64) -> f64 {
		let mut output = 0.0;
		for j in 0..self.exemple_x.len() {
			let diff_x1 = x1 - self.exemple_x[j][0];
			let diff_x2 = x2 - self.exemple_x[j][1];
			let mag_x = diff_x1*diff_x1 + diff_x2*diff_x2;
			output += self.poids[j] * (-self.gammma * mag_x).exp();
		}   
		output
	}
}

#[no_mangle]
extern "C" fn create_rbf(nb_entree: i32, gammma : f64, input : *const f64, rows_in: i32, cols_in: i32, output: *const f64, rows_out: i32) -> *mut Rbf {
	let slice_in = unsafe { std::slice::from_raw_parts(input, (rows_in * cols_in) as usize) };
    let mut input_data = vec![vec![0.0; cols_in as usize]; rows_in as usize];
    for i in 0..rows_in {
        let start = (i * cols_in) as usize;
        let end = ((i + 1) * cols_in) as usize;
        let row = &slice_in[start..end];
        input_data[i as usize] = row.to_vec();
    }
	let output_data = unsafe { std::slice::from_raw_parts(output, rows_out as usize) };
	
	let mut rbf = Box::new(Rbf::new(nb_entree as usize, gammma, input_data, output_data.to_vec()));
	rbf.init_poids();
    Box::leak(rbf)
}

#[no_mangle]
extern "C" fn predict_rbf(ptr: *mut Rbf, input: *const f64, length: i32) -> f64 {
    let model = unsafe { &mut *ptr };
    let slice = unsafe { std::slice::from_raw_parts(input, length as usize) };
    let result = model.predict(slice[0], slice[1]);
	result
}