use rand::Rng;

pub struct MultiLayerPerceptron {
    n_entree : usize,
    n_sortie : usize,
    n_hidden : usize,
    poids : Vec<f64>,
    entrees : Vec<f64>,
    sorties : Vec<f64>,
    n_poids : usize,
}

impl MultiLayerPerceptron {
    fn new() -> MultiLayerPerceptron {
        MultiLayerPerceptron {
            n_entree: 0,
            n_sortie: 0,
            n_hidden: 0,
            poids: Vec::new(),
            entrees: Vec::new(),
            sorties: Vec::new(),
            n_poids: 0,
        }
    }
}


#[no_mangle]
pub extern "C" fn free_mlp(mlp_ptr: *mut MultiLayerPerceptron) {
    unsafe {
        Box::from_raw(mlp_ptr);
    }
}

#[no_mangle]
pub extern "C" fn create_mlp(nb_entree: usize, nb_hidden: usize, nb_sortie: usize) -> *mut MultiLayerPerceptron {
	let mlp = Box::new(init_lin_mod_internal(nb_entree, nb_hidden, nb_sortie));
    Box::into_raw(mlp)
}
	
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
    // Utilisez la fonction init_lin_mod pour initialiser le mlp
    unsafe {
        *mlp_ptr = Box::into_raw(Box::new(init_lin_mod_internal(nb_entree, nb_hidden, nb_sortie)));
    }
}

pub extern "C" fn init_lin_mod_internal(nb_entree: usize, nb_hidden: usize, nb_sortie: usize) -> MultiLayerPerceptron {
    let mut lm = MultiLayerPerceptron {
        n_entree: nb_entree + 1,
        n_sortie: nb_sortie,
        n_hidden: nb_hidden,
        poids: vec![0.0;(nb_entree + 1) * (nb_hidden) + (nb_entree + 1)],
		entrees: vec![0.0;(nb_entree + 1) * (nb_hidden)],
        sorties: vec![0.0; nb_sortie],
        n_poids: (nb_entree + 1) * (nb_hidden) + (nb_entree + 1),
    };
/*
poids: Vec::with_capacity((nb_entree + 1) * (nb_hidden + 1) * (nb_entree + 1)),
        entrees: Vec::with_capacity((nb_entree + 1) * (nb_hidden + 1)),
        sorties: Vec::with_capacity(nb_sortie),*/
	println!("a ");
	println!("entrees {}, sortie {}, poids {} ",lm.entrees.len(), lm.sorties.len(),lm.poids.len());

    for p in 0..lm.n_poids {
        lm.poids[p] = rand::thread_rng().gen_range(-1.0..1.0);
    }
	
	/*
	for i in 0..lm.n_poids {
        println!(" {} ",lm.poids[i]);
    }
	*/
    lm
}

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
	for indexE in 0..mlp.n_entree-1 {
        mlp.entrees[indexE] = test_data[indexE];
		println!("{}",test_data[indexE]);
    }
    //lm.entrees[lm.entrees-1] = 1;

    for indexH in 0..mlp.n_hidden-1 { //Index des couches
        for indexE in 0..mlp.n_entree-1 { //Index de la 2eme colonne de perceptron 
            let mut sum : f64 = 0.0;
            let indexEntree = indexH * mlp.n_entree;
            let indexPoids = indexH * mlp.n_entree + indexE * mlp.n_entree;
			//println!("{} {}", indexEntree, indexPoids);

            for indexN in 0..mlp.n_entree-1 { //Index de la 1ere colonne de perceptron 
                sum += mlp.entrees[indexEntree + indexN] * mlp.poids[indexPoids + indexN];
            }
            sum += mlp.poids[indexPoids + mlp.n_entree-1];
            let res = 1.0/(1.0 + libm::exp(-sum));
			//println!("res {}, index {}", res, (indexH+1) * mlp.n_entree + indexE);
            mlp.entrees[(indexH+1) * mlp.n_entree + indexE] = res;
        }
    }

    for indexS in 0..mlp.n_sortie { //Index de la sortie
        let mut sum : f64 = 0.0;
        let indexEntree = (mlp.n_hidden-1) * mlp.n_entree;
        let indexPoids = (mlp.n_hidden-1) * mlp.n_entree + indexS * mlp.n_entree;
        for indexN in 0..mlp.n_entree-1 { //Index de la 1ere colonne de perceptron 
            sum += mlp.entrees[indexEntree + indexN] * mlp.poids[indexPoids + indexN];
        }
        sum += mlp.poids[indexPoids + mlp.n_entree-1];
        let res = 1.0/(1.0 + libm::exp(-sum));
        mlp.sorties[indexS] = res;
		println!("{}",res);
    }
    mlp
}

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

    // Convertir la référence mutable en pointeur brut pour renvoyer
    //mlp as *mut MultiLayerPerceptron
}

#[no_mangle]
pub extern "C" fn mlpLearning_internal<'a>(mut mlp: &'a mut MultiLayerPerceptron, learn_data: &'a [f64]) {
	let alpha = 0.01;
    let blockSize = mlp.n_entree - 1 + mlp.n_sortie; 
    let nbBlocks = learn_data.len() / blockSize;
    let mut rng = rand::thread_rng();

    for learningTime in 0..10 {
        let randBlockIndex = rng.gen_range(0..nbBlocks-1) * blockSize;
        let mut test_data : &'a [f64] = &learn_data[randBlockIndex..randBlockIndex+mlp.n_entree-1];
        let mut res_data: &'a [f64] = &learn_data[randBlockIndex+mlp.n_entree-1..randBlockIndex+mlp.n_entree+mlp.n_sortie-1];
        mlp = ask_lin_mod_internal(mlp, test_data);
		let res = &mlp.sorties;
        let mut isOk = true;
        for nbSortie in 0..mlp.n_sortie {
            if (res[nbSortie] - res_data[nbSortie]).abs() > 1e-6 {
                isOk = false;
            }
        }
        if !isOk {
            let mut deltas =  vec![0.0;(mlp.n_entree * (mlp.n_hidden + 1) + mlp.n_sortie)];
			//on calcule les deltas de la derniere couche
            for indexE in mlp.n_entree * (mlp.n_hidden-1)..mlp.n_entree * (mlp.n_hidden-1)+mlp.n_entree-1 { //Index de la sortie
                let res = (1.0 - mlp.entrees[indexE]*mlp.entrees[indexE]) * (mlp.entrees[indexE] - res_data[0]);
                deltas[indexE] = res;
            }
			//On calcul le reste des deltas a partir de l avant derniere couche
            for indexH in (0..mlp.n_hidden-1).rev() { //Index des couches
                for indexE in 0..mlp.n_entree { //Index de la 2eme colonne de perceptron 
                    let mut sum : f64 = 0.0;
                    let indexDelta = indexH * mlp.n_entree;
					let indexNextDelta = (indexH+1) * mlp.n_entree;
                    let indexPoids = indexH * mlp.n_entree + indexE * mlp.n_entree;
                    for indexN in 0..mlp.n_entree { //Index de la 1ere colonne de perceptron 
                        sum += deltas[indexNextDelta + indexN] * mlp.poids[indexPoids + indexN];
                    }
                    let res = (1.0 - mlp.entrees[indexE]*mlp.entrees[indexE]) * sum;
                    deltas[indexDelta] = res;
                }
            }
			//On met a jour les poids
			println!("timer : {} ",learningTime); 
			for indexH in 0..mlp.n_hidden-1 {
                for indexI in 0..mlp.n_entree {
                    for indexJ in 0..mlp.n_entree {
                        let indexPoids = indexH * mlp.n_entree * mlp.n_entree + indexI * mlp.n_entree + indexJ;
                        let indexEntree = indexH * mlp.n_entree + indexI;
                        let indexDelta = indexH * mlp.n_entree + indexJ;
                        mlp.poids[indexPoids] = mlp.poids[indexPoids] - alpha * mlp.entrees[indexEntree] * deltas[indexDelta];
						println!("nb poids {}, val  {} ",indexPoids,deltas[indexDelta]);
                    }
                }
            }
        }
    }
}