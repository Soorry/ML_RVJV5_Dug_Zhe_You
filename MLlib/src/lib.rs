use rand::Rng;

struct MultiLayerPerceptron {
    n_entree : usize,
    n_sortie : usize,
    n_hidden : usize,
    poids : Vec<f64>,
    entrees : Vec<f64>,
    sorties : Vec<f64>,
    n_poids : usize,
}

static mut mlp : MultiLayerPerceptron = MultiLayerPerceptron {
	n_entree : 0,
	n_sortie : 0,
	n_hidden : 0,
	poids : Vec::new(),
	entrees : Vec::new(),
	sorties : Vec::new(),
	n_poids : 0,
};
	
#[no_mangle]
pub extern "C" fn main(nb_entree : usize, nb_hidden : usize, nb_sortie : usize) -> &'static MultiLayerPerceptron {
    println!("start");
	unsafe {
		mlp = init_lin_mod(nb_entree, nb_hidden, nb_sortie);
		&mlp
	}
}

pub extern "C" fn init_lin_mod(nb_entree : usize, nb_hidden : usize, nb_sortie : usize) -> MultiLayerPerceptron {
    let mut lm = MultiLayerPerceptron {
        n_entree : nb_entree+1,
        n_sortie : nb_sortie,
        n_hidden : nb_hidden,
        poids : Vec::with_capacity((nb_entree +1) * (nb_hidden + 1) * (nb_entree +1)),
        entrees : Vec::with_capacity((nb_entree +1) * (nb_hidden + 1)),
        sorties : Vec::with_capacity(nb_sortie),
        n_poids : (nb_entree+1)*nb_sortie+nb_hidden*(nb_entree+1)*(nb_entree+1),
    };

    for index in 0..lm.n_poids {
        lm.poids.push(rand::thread_rng().gen_range(-1.0..1.0));
    }

    lm
}

pub extern "C" fn ask_lin_mod(mut lm : &mut MultiLayerPerceptron,test_data : Vec<f64>) -> &mut MultiLayerPerceptron {
    for indexE in 0..lm.n_entree-1 {
        lm.entrees[indexE] = test_data[indexE];
    }
    //lm.entrees[lm.entrees-1] = 1;

    for indexH in 0..lm.n_hidden { //Index des couches
        for indexE in 0..lm.n_entree { //Index de la 2eme colonne de perceptron 
            let mut sum : f64 = 0.0;
            let indexEntree = indexH * lm.n_entree;
            let indexPoids = indexH * lm.n_entree + indexE * lm.n_entree;
            for indexN in 0..lm.n_entree-1 { //Index de la 1ere colonne de perceptron 
                sum += lm.entrees[indexEntree + indexN] * lm.poids[indexPoids + indexN];
            }
            sum += lm.poids[indexPoids + lm.n_entree-1];
            let res = 1.0/(1.0 + libm::exp(-sum));
            lm.entrees[(indexH+1) * lm.n_entree] = res;
        }
    }

    for indexS in 0..lm.n_sortie { //Index de la sortie
        let mut sum : f64 = 0.0;
        let indexEntree = lm.n_hidden * lm.n_entree;
        let indexPoids = lm.n_hidden * lm.n_entree + indexS * lm.n_entree;
        for indexN in 0..lm.n_entree-1 { //Index de la 1ere colonne de perceptron 
            sum += lm.entrees[indexEntree + indexN] * lm.poids[indexPoids + indexN];
        }
        sum += lm.poids[indexPoids + lm.n_entree-1];
        let res = 1.0/(1.0 + libm::exp(-sum));
        lm.sorties[indexS] = res;
		println!('{}',res);
    }
    lm
}

pub extern "C" fn mlpLearning(mut lm : &mut MultiLayerPerceptron,learn_data : Vec<f64>) {
	let alpha = 0.01;
    let blockSize = lm.n_entree + lm.n_sortie; 
    let nbBlocks = blockSize / learn_data.len();
    let mut rng = rand::thread_rng();
    for learningTime in 0..10 {
        let randBlockIndex = rng.gen_range(0..nbBlocks-1) * blockSize;
        let mut test_data : Vec<f64> = Vec::new();
        for nbEntree in randBlockIndex..randBlockIndex+lm.n_entree {
            test_data.push(learn_data[nbEntree]);
        }
        let mut res_data : Vec<f64> = Vec::new();
        for nbSortie in randBlockIndex+lm.n_entree..randBlockIndex+lm.n_entree+lm.n_sortie {
            res_data.push(learn_data[nbSortie]);
        }
        lm = ask_lin_mod(lm, test_data);
		let res = &lm.sorties;
        let mut isOk = true;
        for nbSortie in 0..lm.n_sortie {
            if (res[nbSortie] - res_data[nbSortie]).abs() > 1e-6 {
                isOk = false;
            }
        }
        if !isOk {
            let mut deltas: Vec<f64> = Vec::with_capacity(lm.n_entree * (lm.n_hidden + 1) + lm.n_sortie);
			//on calcule les deltas de la derniere couche
            for indexS in 0..lm.n_sortie { //Index de la sortie
                let res = (1.0 - lm.sorties[indexS]*lm.sorties[indexS]) * (lm.sorties[indexS] - res_data[lm.n_sortie]);
                deltas[lm.n_entree * (lm.n_hidden + 1) + indexS] = res;
            }
			//On calcul le reste des deltas a partir de l avant derniere couche
            for indexH in (0..lm.n_hidden).rev() { //Index des couches
                for indexE in 0..lm.n_entree { //Index de la 2eme colonne de perceptron 
                    let mut sum : f64 = 0.0;
                    let indexDelta = indexH * lm.n_entree;
					let indexNextDelta = (indexH+1) * lm.n_entree;
                    let indexPoids = indexH * lm.n_entree + indexE * lm.n_entree;
                    for indexN in 0..lm.n_entree { //Index de la 1ere colonne de perceptron 
                        sum += deltas[indexNextDelta + indexN] * lm.poids[indexPoids + indexN];
                    }
                    let res = (1.0 - lm.sorties[indexE]*lm.sorties[indexE]) * sum;
                    lm.entrees[indexDelta] = res;
                }
            }
			//On met a jour les poids
			for indexH in 0..lm.n_hidden {
                for indexI in 0..lm.n_entree {
                    for indexJ in 0..lm.n_entree {
                        let indexPoids = indexH * lm.n_entree * lm.n_entree + indexI * lm.n_entree + indexJ;
                        let indexEntree = indexH * lm.n_entree + indexI;
                        let indexDelta = indexH * lm.n_entree + indexJ;
                        lm.poids[indexPoids] = lm.poids[indexPoids] - alpha * lm.entrees[indexEntree] * deltas[indexDelta];
                    }
                }
            }
        }
    }
}