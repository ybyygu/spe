// [[file:~/Workspace/Programming/spe/spe.note::b3eec740-ffac-473e-bc66-bbeb404bf7b5][b3eec740-ffac-473e-bc66-bbeb404bf7b5]]
#[macro_use]
extern crate approx;

extern crate rand;
extern crate itertools;

type Point3D = [f64; 3];
type Points = Vec<Point3D>;

const EPSILON: f64 = 1.0E-6;

/// Update the coordinates pi and pj
/// Parameters
/// ----------
/// pi, pj: the positions of a pair of particles in 3D space
/// lij   : lower bound of the target distance between pi and pj
/// uij   : upper bound of target distance between pi and pj
/// lam   : a parameter defining learning rate
///
fn update_coordinates_of_pair(pi: &mut Point3D, pj: &mut Point3D, lij: f64, uij: f64, lam: f64) {
    let dx = pj[0] - pi[0];
    let dy = pj[1] - pi[1];
    let dz = pj[2] - pi[2];
    let dij = (dx*dx + dy*dy + dz*dz).sqrt();

    debug_assert!(uij >= lij);
    if dij >= lij && dij <= uij {
        return;
    }

    let rij = if (dij - lij).abs() < (dij - uij).abs() {
        uij
    } else {
        lij
    };

    let disparity = lam*(rij - dij) / (dij + EPSILON);
    for v in 0..3 {
        pi[v] += 0.5 * disparity * (pi[v] - pj[v]);
        pj[v] += 0.5 * disparity * (pj[v] - pi[v]);
    }
}

#[test]
fn test_update_coordinates() {
    let mut pi = [7.24625784,   6.75592996,  11.23685344];
    let mut pj = [8.71022084,  11.06181612,   9.59290676];
    update_coordinates_of_pair(&mut pi, &mut pj, 1.8, 1.8, 0.8);

    let rpi = [  7.61388092,   7.83720259,  10.82403376];
    let rpj = [  8.43491343,  10.25206727,   9.90206122];
    for v in 0..3 {
        assert_relative_eq!(pi[v], rpi[v], epsilon=1e-4);
        assert_relative_eq!(pj[v], rpj[v], epsilon=1e-4);
    }
}
// b3eec740-ffac-473e-bc66-bbeb404bf7b5 ends here

// [[file:~/Workspace/Programming/spe/spe.note::4edc3f04-fac1-48fa-bc4e-258bf982a1ef][4edc3f04-fac1-48fa-bc4e-258bf982a1ef]]
use rand::{thread_rng, Rng};

/// the isometric variant of Stochastic Proximity Embedding
///
/// References
/// ----------
/// (1) Agrafiotis, D. K. J. Comput. Chem. 2003, 24 (10), 1215–1221. (SPE)
/// (2) Agrafiotis, D. K.; Xu, H. J. Chem. Inf. Comput. Sci. 2003, 43 (2), 475–484. (ISPE)
///
fn ispe(points: &mut Points, bounds: Vec<Vec<f64>>, maxcycle: usize, maxstep: usize, maxlam: f64, minlam: f64) {
    assert!(maxlam > minlam, "max learning rate is smaller than min learning rate.");
    assert!(minlam > 0.0, "learning rate cannot be smaller than zero.");

    let mut lam = maxlam;

    // create a list of adjustable pairs
    let mut possible_pairs = vec![];
    let npoints = points.len();
    for i in 0..npoints {
        for j in 0..npoints {
            if i != j {
                possible_pairs.push((i, j));
            }
        }
    }

    // update the positions of pairs iteratively
    let mut rng = thread_rng();
    let dlam = (maxlam - minlam)/(maxcycle as f64);
    for icycle in 0..maxcycle {
        lam -= dlam;
        for istep in 0..maxstep {
            let &(i, j) = rng.choose(&possible_pairs).unwrap();
            let mut pi = points[i];
            let mut pj = points[j];
            let (lij, uij) = target_distances(&bounds, i, j);
            update_coordinates_of_pair(&mut pi, &mut pj, lij, uij, lam);
            // set back
            points[i] = pi;
            points[j] = pj;
        }
    }
}

use std::mem;
/// get the lower and upper bound of target distances for a pair of points
fn target_distances(bounds: &Vec<Vec<f64>>, i: usize, j: usize) -> (f64, f64) {
    // make sure i < j
    let mut i = i;
    let mut j = j;
    if i > j {
        mem::swap(&mut i, &mut j);
    }

    let lij = bounds[i][j];
    let uij = bounds[j][i];

    (lij, uij)
}
// 4edc3f04-fac1-48fa-bc4e-258bf982a1ef ends here

// [[file:~/Workspace/Programming/spe/spe.note::4432eb02-aa74-459f-9658-29bc3a7f8ef5][4432eb02-aa74-459f-9658-29bc3a7f8ef5]]
/// The pivoted variant of Stochastic Proximity Embedding (PSPE)
///
/// References
/// ----------
/// (1) J. Mol. Graph. Model. 2003, 22 (2), 133–140.
/// (2) J. Chem. Inf. Model. 2011, 51 (11), 2852–2859.
///
fn pspe(points: &mut Points, bounds: Vec<Vec<f64>>, maxcycle: usize, maxstep: usize, maxlam: f64, minlam: f64) {
    assert!(maxlam > minlam, "max learning rate is smaller than min learning rate.");
    assert!(minlam > 0.0, "learning rate cannot be smaller than zero.");

    let mut lam = maxlam;

    // create a list of adjustable pairs
    let mut possible_pairs = vec![];
    let npoints = points.len();
    for i in 0..npoints {
        for j in 0..npoints {
            if i != j {
                possible_pairs.push((i, j));
            }
        }
    }

    let indices: Vec<_> = (0..npoints).collect();
    // update the positions of pairs iteratively
    let dlam = (maxlam - minlam)/(maxcycle as f64);
    let mut rng = thread_rng();
    for icycle in 0..maxcycle {
        lam -= dlam;
        // index of the pivot point
        let &i = rng.choose(&indices).unwrap();
        let mut pi = points[i];
        for istep in 0..maxstep {
            let mut j = i;
            while j == i {
                j = *rng.choose(&indices).unwrap();
            };

            // get lower and upper bounds for target distance
            let (lij, uij) = target_distances(&bounds, i, j);

            let mut pj = points[j];
            update_coordinates_of_pair(&mut pi, &mut pj, lij, uij, lam);
            // set back
            points[i] = pi;
            points[j] = pj;
        }
    }
}
// 4432eb02-aa74-459f-9658-29bc3a7f8ef5 ends here
