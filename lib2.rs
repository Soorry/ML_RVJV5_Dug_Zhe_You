use rand::Rng;
use audio_visualizer::waveform::png_file::waveform_static_png_visualize;
use audio_visualizer::ChannelInterleavement;
use audio_visualizer::Channels;
use minimp3::{Decoder as Mp3Decoder, Error as Mp3Error, Frame as Mp3Frame};
use std::fs::File;
use std::path::PathBuf;

struct MultiLayerPerceptron {
    n_entree : usize,
    n_sortie : usize,
    n_couche : usize,
    poids : Vec<Vec<Vec<f64>>>,
    entrees : Vec<Vec<f64>>,
    sorties : Vec<f64>,
	deltas : Vec<Vec<f64>>,
    n_poids : usize,
}

impl MultiLayerPerceptron {
    pub fn new(nb_entree: usize, nb_couche: usize, nb_sortie: usize) -> Self {
        Self {
			n_entree: nb_entree + 1,
			n_sortie: nb_sortie,
			n_couche: nb_couche ,
			poids: vec![vec![vec![-1.0;(nb_entree + 1)]; (nb_entree + 1)]; nb_couche],
			entrees: vec![vec![1.0;nb_entree + 1]; nb_couche],
			sorties: vec![0.0; nb_sortie],
			deltas: vec![vec![0.0;nb_entree + 1]; nb_couche],
			n_poids: (nb_entree + 1) * (nb_couche) + (nb_entree + 1)
        }
    }
	
	pub fn initPoids(&mut self) {
		for l in 0..self.n_couche {
			for i in 0..self.n_entree {
				for j in 0..self.n_entree {
					self.poids[l][i][j] = rand::thread_rng().gen_range(-1.0..1.0);
				}
			}
		}
	}
	
	pub fn predict(&mut self, input : Vec<f64>) -> Vec<f64>  {
		//On ajoute les entrées au MLP
		for i in 0..input.len() {
			self.entrees[0][i] = input[i];
		}

		//On calcul les output intermédiaire pour chaque couche
		for l in 0..self.n_couche-1 { //Index des couches
			for j in 0..self.n_entree-1 { //Index de la 2eme colonne de perceptron 
				let mut sum : f64 = 0.0;
				for i in 0..self.n_entree { //Index de la 1ere colonne de perceptron 
					sum += self.entrees[l][i] * self.poids[l][i][j];
				}
				let res = sum.tanh();
				self.entrees[l+1][j] = res;
			}
		}
		
		for i in 0..self.n_sortie {
            self.sorties[i] = self.entrees[self.n_couche-1][i];
        }
		
		(&self.sorties).to_vec()
	}
	
	pub fn train(&mut self, input: Vec<Vec<f64>>, output: Vec<Vec<f64>>) {
		let alpha = 0.01;
		let max_iter = 1000;

		let mut rng = rand::thread_rng();
		for _learning_time in 0..max_iter {
			println!("err : {}", self.calculate_loss(&input, &output));			
			// Choisissez un exemple au hasard
			let rand_data_index = rng.gen_range(0..input.len());
			// Séparez les output et les input
			let input_data = &input[rand_data_index];
			let output_data = &output[rand_data_index];

			// Passez l'exemple dans le MLP
			let _out = self.predict(input_data.to_vec());
			
			if (_out[0] - output_data[0]).abs() < 0.1 {
				continue;
			}
			// Calculate deltas for the output layer
			for j in 0..self.n_sortie {
				self.deltas[self.n_couche - 1][j] =
					self.entrees[self.n_couche - 1][j] - output_data[j];
				if true {
					self.deltas[self.n_couche - 1][j] *=
						(1.0 - self.entrees[self.n_couche - 1][j].powi(2));
				}
			}

			// Calculate deltas for hidden layers
			for l in (1..self.n_couche - 1).rev() {
				for i in 0..self.n_entree {
					let mut sum = 0.0;
					for j in 0..self.n_entree {
						sum += self.poids[l][i][j] * self.deltas[l][j];
					}
					self.deltas[l][i] =
						(1.0 - self.entrees[l][i].powi(2)) * sum;
				}
			}

			// Update weights using deltas and learning rate (alpha)
			for l in 1..self.n_couche {
				for i in 0..self.n_entree {
					for j in 0..self.n_entree {
						self.poids[l][i][j] -= alpha
							* self.entrees[l][i]
							* self.deltas[l][j];
					}
				}
			}
		}
	}

	
	pub fn train2(&mut self, input: Vec<Vec<f64>>, output: Vec<Vec<f64>>) {
		let alpha = 0.1;
		let maxIter = 100;

		let mut rng = rand::thread_rng();
		for learningTime in 0..maxIter {
			//On choisis un exemple au hasard
			let randDataIndex = rng.gen_range(0..input.len());
			//On sépare les output et les input
			let mut input_data = &input[randDataIndex];
			let mut output_data = &output[randDataIndex];
			println!("err : {}", self.calculate_loss(&input, &output));			
			//On passe l'exemple dans le mlp
			let out = self.predict(input_data.to_vec());
			let mut isOk = true;
			//On check si la sortie est correct
			for nbSortie in 0..self.n_sortie {
				if (out[nbSortie] - output_data[nbSortie]).abs() > 1e-6 {
					isOk = false;
				}
			}
			if !isOk {
				// Calculate deltas for the output layer

				for j in 0..self.n_sortie {
					self.deltas[self.n_couche-1][j] = self.entrees[self.n_couche-1][j] - output_data[j];
					if true {
						self.deltas[self.n_couche-1][j] *= (1.0 - self.entrees[self.n_couche-1][j].powi(2));
					}
				}

				// Calculate deltas for hidden layers
				for l in (1..self.n_couche).rev() {
					for i in 0..self.n_entree {
						let mut sum = 0.0;
						for j in 0..self.n_entree {
							sum += self.poids[l][i][j] * self.deltas[l][j];
						}
						self.deltas[l - 1][i] = (1.0 - self.entrees[l - 1][i].powi(2)) * sum;
					}
				}

				// Update weights using deltas and learning rate (alpha)
				for l in 0..self.n_couche {
					for i in 0..self.n_entree {
						for j in 0..self.n_entree {
							self.poids[l][i][j] -= alpha * self.entrees[l][i] * self.deltas[l][j];
							//println!("poids  {}", self.poids[l][i][j]);
						}
					}
				}
				println!("---");

				/*
				//Si ce n'est pas le cas on corrige
				let mut deltas =  vec![vec![1.0;self.n_entree]; self.n_couche];

				//on calcule les deltas de la derniere couche
				for l in 0..self.n_sortie { //Index de la sortie
					let res = (1.0 - self.entrees[&self.n_couche-1][l]*self.entrees[&self.n_couche-1][l]) * (self.entrees[&self.n_couche-1][l] - output_data[l]);
					deltas[&self.n_couche-1][l] = res;
					println!("delta der  {}", res);
				}
				//On calcul le reste des deltas a partir de l avant derniere couche
				for l in (1..self.n_couche).rev() { //Index des couches
					for i in 1..self.n_entree { //Index de la 2eme colonne de perceptron 
						let mut sum : f64 = 0.0;
						for j in 0..self.n_entree { //Index de la 1ere colonne de perceptron 
							sum += deltas[l][j] * self.poids[l][i][j];
						}
						let res = (1.0 - self.entrees[l-1][i]*self.entrees[l-1][i]) * sum;
						deltas[l-1][i] = res;
						println!("resd  {}", res);
					}
				}

				for l in 1..self.n_couche {
					for i in 0..self.n_entree {
						for j in 0..self.n_entree {
							self.poids[l][i][j] -= (alpha * self.entrees[l-1][i] * deltas[l][j]);
							//println!("poids  {}", self.poids[l][i][j]);
						}
					}
				}
				
				println!("---");
				*/
				/*
				//Affichage
				for l in 0..self.n_couche {
					for i in 0..self.n_entree {
						println!("e:{}, d:{}",self.entrees[l][i],deltas[l][i]);
					}
				}*/
				
				//self.stochastic_gradient_descent(input_data, output_data, alpha);
			}
		}
	}
	
	// Ajouter la fonction pour effectuer la rétropropagation du gradient stochastique
	fn stochastic_gradient_descent(&mut self, input: &Vec<f64>, target: &Vec<f64>, learning_rate: f64) {
		// Effectuer une prédiction avec l'entrée actuelle
		let prediction = self.predict(input.to_vec());

		// Calculer les erreurs
		let errors: Vec<f64> = prediction.iter().zip(target.iter()).map(|(p, t)| t - p).collect();

		// Calculer les gradients pour chaque poids en utilisant la rétropropagation du gradient
		let gradients = self.calculate_gradients(input, errors);

		// Mettre à jour les poids en fonction des gradients et du taux d'apprentissage
		self.update_weights(gradients, learning_rate);
	}

	// Ajouter la fonction pour calculer les gradients lors de la rétropropagation du gradient
	fn calculate_gradients(&mut self, input: &Vec<f64>, mut errors: Vec<f64>) -> Vec<Vec<Vec<f64>>> {
		// Initialiser une structure pour stocker les gradients
		let mut gradients: Vec<Vec<Vec<f64>>> = vec![vec![vec![0.0; self.n_entree]; self.n_entree]; self.n_couche];

		// Calculer les gradients pour chaque poids en utilisant la rétropropagation du gradient
		for c in (0..self.n_couche-1).rev() {
			for i in 0..self.n_entree {
				for j in 0..self.n_entree {
					if(j==self.n_entree-1)
					{
						gradients[c][i][j] = errors[i] * self.derivative_tanh(self.entrees[c + 1][i]);
					}
					else
					{
						gradients[c][i][j] = errors[i] * self.derivative_tanh(self.entrees[c + 1][i]) * input[j];
					}
				}
			}
			
			let mut next_errors: Vec<f64> = vec![0.0; self.n_entree];
			// Calculer les erreurs pour la couche précédente
			for i in 0..self.n_entree {
				for j in 0..self.n_entree {
					next_errors[i] += errors[j] * self.poids[c][j][i];
				}
			}

			errors = next_errors;
		}

		gradients
	}

	// Ajouter la fonction pour mettre à jour les poids en fonction des gradients et du taux d'apprentissage
	fn update_weights(&mut self, gradients: Vec<Vec<Vec<f64>>>, learning_rate: f64) {
		for c in 0..self.n_couche {
			for i in 0..self.n_entree {
				for j in 0..self.n_entree {
					self.poids[c][i][j] += learning_rate * gradients[c][i][j];
				}
			}
		}
	}

	// Ajouter une fonction dérivée pour tanh
	fn derivative_tanh(&self, x: f64) -> f64 {
		1.0 - x.tanh().powi(2)
	}
	
	fn calculate_loss(&mut self, inputs: &Vec<Vec<f64>>, targets: &Vec<Vec<f64>>) -> f64 {
		let mut total_loss = 0.0;
		for (input, target) in inputs.iter().zip(targets.iter()) {
			let prediction = self.predict(input.to_vec());
			println!("{}", prediction[0]);
			let errors: Vec<f64> = prediction.iter().zip(target.iter()).map(|(p, t)| t - p).collect();
			total_loss += errors.iter().map(|e| e.powi(2)).sum::<f64>();
		}

		total_loss / (inputs.len() as f64)
	}
}

/*
#[no_mangle]
pub extern "C" fn free_mlp(mlp_ptr: *mut MultiLayerPerceptron) {
    unsafe {
        Box::from_raw(mlp_ptr);
    }
}
*/


#[no_mangle]
pub extern "C" fn train(ptr: *mut MultiLayerPerceptron, input : *const f64, rowsIn: i32, colsIn: i32, output: *const f64, rowsOut: i32, colsOut: i32) {
    let model = unsafe { &mut *ptr };
    let sliceIn = unsafe { std::slice::from_raw_parts(input, (rowsIn * colsIn) as usize) };
    let mut input_data = vec![vec![0.0; colsIn as usize]; rowsIn as usize];
    for i in 0..rowsIn {
        let start = (i * colsIn) as usize;
        let end = ((i + 1) * colsIn) as usize;
        let row = &sliceIn[start..end];
        input_data[i as usize] = row.to_vec();
    }

	let sliceOut = unsafe { std::slice::from_raw_parts(output, (rowsOut * colsOut) as usize) };
    let mut output_data =  vec![vec![0.0; colsOut as usize]; rowsOut as usize];
    for i in 0..rowsOut {
        let start = (i * colsOut) as usize;
        let end = ((i + 1) * colsOut) as usize;
        let row = &sliceOut[start..end];
        output_data[i as usize] = row.to_vec();
    }

    model.train(input_data, output_data);
}

#[no_mangle]
pub extern "C" fn create_mlp(nb_entree: i32, nb_couche: i32, nb_sortie: i32) -> *mut MultiLayerPerceptron {
	let mut mlp = Box::new(MultiLayerPerceptron::new(nb_entree as usize, nb_couche as usize, nb_sortie as usize));
	mlp.initPoids();
    Box::leak(mlp)
}

#[no_mangle]
pub extern "C" fn predict(ptr: *mut MultiLayerPerceptron, input: *const f64, length: i32) -> *mut f64 {
    let model = unsafe { &mut *ptr };
    let slice = unsafe { std::slice::from_raw_parts(input, length as usize) };
    let result = model.predict(slice.to_vec());
    result.leak().as_mut_ptr()
}

	/*
#[no_mangle]
pub extern "C" fn main(nb_entree: usize, nb_hidden: usize, nb_sortie: usize) -> *mut MultiLayerPerceptron {
    // Créez une nouvelle instance de MultiLayerPerceptron
    let mut mlp = MultiLayerPerceptron::new();

    // Utilisez la fonction init_lin_mod pour initialiser le mlp
    let mut mlp_ptr: *mut MultiLayerPerceptron = &mut mlp;
	init_lin_mod(&mut mlp_ptr, nb_entree, nb_hidden, nb_sortie);

    // Retournez le pointeur vers le MultiLayerPerceptron initialisé
    mlp_ptr
}

#[no_mangle]
pub extern "C" fn init_lin_mod(mlp_ptr: *mut *mut MultiLayerPerceptron, nb_entree: usize, nb_hidden: usize, nb_sortie: usize) {
    unsafe {
        *mlp_ptr = Box::into_raw(Box::new(init_lin_mod_internal(nb_entree, nb_hidden, nb_sortie)));
    }
}

pub extern "C" fn init_lin_mod_internal(nb_entree: usize, nb_couche: usize, nb_sortie: usize) -> MultiLayerPerceptron {
    let mut lm = MultiLayerPerceptron {
        n_entree: nb_entree + 1,
        n_sortie: nb_sortie,
        n_couche: nb_couche,
        poids: vec![vec![vec![0.0;(nb_entree + 1)]; (nb_entree + 1)]; nb_couche],
		entrees: vec![vec![0.0;nb_entree + 1]; nb_couche],
        sorties: vec![0.0; nb_sortie],
        n_poids: (nb_entree + 1) * (nb_couche) + (nb_entree + 1),
    };

	//println!("entrees {}, sortie {}, poids {} ",lm.entrees.len(), lm.sorties.len(),lm.poids.len());

	//Création des poids de façon aléatoire
    for l in 0..lm.n_couche {
		for i in 0..lm.n_entree {
			for j in 0..lm.n_entree {
				lm.poids[l][i][j] = rand::thread_rng().gen_range(-1.0..1.0);
			}
		}
    }
	
	//Test avec les poids du XOR
	// lm.poids[0] = 1.0;
	// lm.poids[1] = 1.0;
	// lm.poids[2] = -0.5;
	// lm.poids[3] = -1.0;
	// lm.poids[4] = -1.0;
	// lm.poids[5] = 1.5;
	// lm.poids[6] = 1.0;
	// lm.poids[7] = 1.0;
	// lm.poids[8] = -1.5;
	
	/*
	for i in 0..lm.n_poids {
        println!(" {} ",lm.poids[i]);
    }
	*/
    lm
}

//Fonction utilisée pour être appelé par le python
#[no_mangle]
pub extern "C" fn ask_lin_mod<'a>(
    mlp: *mut MultiLayerPerceptron,
    test_data: *mut f64,
    data_len: usize,
) -> *mut MultiLayerPerceptron { 
	// Convertir le pointeur brut vers la structure Rust
    let mut mlp = unsafe { &mut *mlp };

    // Convertir le pointeur brut vers les données f64 en un slice
    let test_data_slice = unsafe { std::slice::from_raw_parts(test_data, data_len) };

    // Appeler la fonction avec les paramètres convertis
    mlp = ask_lin_mod_internal(mlp, test_data_slice);

    // Convertir la référence mutable en pointeur brut pour renvoyer
    mlp as *mut MultiLayerPerceptron
}

#[no_mangle]
fn ask_lin_mod_internal<'a>(
    mlp: &'a mut MultiLayerPerceptron,
    test_data: &'a [f64],
) -> &'a mut MultiLayerPerceptron {
	
	//On ajoute les entrées au MLP
	for i in 1..mlp.n_entree {
        mlp.entrees[0][i] = test_data[i-1];
    }

	//On calcul les output intermédiaire pour chaque couche
    for l in 1..mlp.n_couche { //Index des couches
        for j in 1..mlp.n_entree { //Index de la 2eme colonne de perceptron 
            let mut sum : f64 = 0.0;
			//let indexEntree = indexH * mlp.n_entree;
            //let indexPoids = indexH * mlp.n_entree + indexE * mlp.n_entree;
			//println!("{} {}", indexEntree, indexPoids);

            for i in 0..mlp.n_entree { //Index de la 1ere colonne de perceptron 
                sum += mlp.entrees[l-1][i] * mlp.poids[l][i][j];
            }
            //sum += mlp.poids[indexPoids + mlp.n_entree-1];
            let res = sum.tanh();
			//println!("res {}, index {}", res, (indexH+1) * mlp.n_entree + indexE);
            mlp.entrees[l][j] = res;
        }
    }
    mlp
}

//Fonction utilisée pour être appelé par le python
#[no_mangle]
pub extern "C" fn mlpLearning<'a>(
    mlp: *mut MultiLayerPerceptron,
    test_data: *mut f64,
    data_len: usize)
{
	// Convertir le pointeur brut vers la structure Rust
    let mut mlp = unsafe { &mut *mlp };

    // Convertir le pointeur brut vers les données f64 en un slice
    let test_data_slice = unsafe { std::slice::from_raw_parts(test_data, data_len) };

    // Appeler la fonction avec les paramètres convertis
    mlpLearning_internal(mlp, test_data_slice);
}

//Entrainement du MLP
#[no_mangle]
pub extern "C" fn mlpLearning_internal<'a>(mut mlp: &'a mut MultiLayerPerceptron, learn_data: &'a [f64]) {
	let alpha = 0.1;
	let maxIter = 1000;
	//La structure utilisé est un seul tableau avec les input et output les un à la suite des autres
	//Il faut donc calculer la taille et le nombre des blocs en fonctions du nombre d'entrée et de sortie
    let blockSize = mlp.n_entree - 1 + mlp.n_sortie;
    let nbBlocks = learn_data.len() / blockSize;
	
    let mut rng = rand::thread_rng();
    for learningTime in 0..maxIter {
		//On choisis un exemple au hasard
        let randBlockIndex = rng.gen_range(0..nbBlocks-1) * blockSize;
		//On sépare les output et les input
        let mut test_data : &'a [f64] = &learn_data[randBlockIndex..randBlockIndex+mlp.n_entree-1];
        let mut res_data: &'a [f64] = &learn_data[randBlockIndex+mlp.n_entree-1..randBlockIndex+mlp.n_entree+mlp.n_sortie-1];
		//On passe l'exemple dans le mlp
        mlp = ask_lin_mod_internal(mlp, test_data);
		// println!("expect : {}",res_data[0]);
		// println!(" ");
		let res = &mlp.sorties;
        let mut isOk = true;
		//On check si la sortie est correct
        for nbSortie in 0..mlp.n_sortie {
            if (res[nbSortie] - res_data[nbSortie]).abs() > 1e-6 {
                isOk = false;
            }
        }
        if !isOk {
			//Si ce n'est pas le cas on corrige
            let mut deltas =  vec![vec![0.0;mlp.n_entree]; mlp.n_couche];

			//on calcule les deltas de la derniere couche
            for l in 0..mlp.n_sortie { //Index de la sortie
                let res = (1.0 - mlp.entrees[&mlp.n_couche-1][l]*mlp.entrees[&mlp.n_couche-1][l]) * (mlp.entrees[&mlp.n_couche-1][l] - res_data[l]);
                deltas[&mlp.n_couche-1][l] = res;
				println!("delta nb : {}, resd  {}", 3, res);
            }
			//On calcul le reste des deltas a partir de l avant derniere couche
            for l in (1..mlp.n_couche-1).rev() { //Index des couches
                for i in 1..mlp.n_entree { //Index de la 2eme colonne de perceptron 
                    let mut sum : f64 = 0.0;
                    for j in 0..mlp.n_entree { //Index de la 1ere colonne de perceptron 
                        sum += deltas[l][j] * mlp.poids[l][i][j];
                    }
                    let res = (1.0 - mlp.entrees[l-1][i]*mlp.entrees[l-1][i]) * sum;
                    deltas[l-1][i] = res;
					//println!("delta nb : {}, resd  {}", indexDelta, res);
                }
            }
			
			// for i in 0..deltas.len() {
				// println!("i {} : {}",i, deltas[i]);
			// }
			
			//On met a jour les poids
			// println!("timer : {} ",learningTime); 
			for l in 1..mlp.n_couche-1 {
                for i in 0..mlp.n_entree {
                    for j in 0..mlp.n_entree {
                        //let indexPoids = indexH * mlp.n_entree * mlp.n_entree + indexI * mlp.n_entree + indexJ;
                        //let indexEntree = indexH * mlp.n_entree + indexI;
                        //let mut indexDelta = indexI;
						//if(indexH > 0) {
							//indexDelta = 3;
						//}
                        mlp.poids[l][i][j] = mlp.poids[l][i][j] - alpha * mlp.entrees[l-1][i] * deltas[l][j];
                    }
                }
            }
        }
    }
}*/