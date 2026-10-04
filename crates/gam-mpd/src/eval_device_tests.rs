//! The device decoder against `counterfactual`'s CPU decoder on the host reference backend (and
//! every float64 accelerator the process has): native and explained programs, every action, and
//! runs started from the clean run's row and layer.

use super::counterfactual::{Action, Decoder, Donor as HostDonor, InputChange, Library, Maps as HostMaps, OutputChange, Program, Rows, Selection, Selector, site_index};
use super::eval_device::{Clean, Donor, GivenRows, Maps, Resident, Run, Units};
use super::explanation_tests::tiny_export;
use gam_gpu::tensor::{Arithmetic, Device};
use ndarray::Array2;
use std::sync::Arc;

fn devices() -> Vec<Device> {
    let mut out = vec![Device::host()];
    if let Ok(Some(d)) = Device::accelerator(gam_gpu::GpuPolicy::Auto) {
        out.push(d);
    }
    out
}

fn gap(a: &Array2<f64>, b: &Array2<f64>) -> f64 {
    assert_eq!(a.dim(), b.dim());
    (a - b).iter().fold(0.0_f64, |m, x| m.max(x.abs()))
}

/// Each site's exact library of input-coordinate units, `v_c = e_c`, `u_c = W e_c`, half of them
/// on per row by a fixed pattern.
fn coordinate(decoder: &Decoder, rows: usize) -> (Vec<Option<Library>>, Selection) {
    let libraries: Vec<Option<Library>> = (0..decoder.sites())
        .map(|site| {
            let w = decoder.native(site);
            Some(Library { v: Array2::eye(w.ncols()), u: w.t().to_owned() })
        })
        .collect();
    let selection = Selection {
        rows: (0..rows)
            .map(|r| libraries.iter().enumerate().flat_map(|(k, l)| (0..l.as_ref().map_or(0, |l| l.v.nrows())).filter(move |c| (r + c + k) % 3 != 0).map(move |c| (k as u32, c as u32))).collect())
            .collect(),
    };
    (libraries, selection)
}

fn host_donor(decoder: &Decoder, maps: HostMaps<'_>, tokens: &[u32], keys: &[(usize, usize, bool)]) -> HostDonor {
    let mut program = Program::new(maps, &[], None);
    program.record = keys.iter().map(|k| (*k, None)).collect();
    decoder.forward(tokens, &mut program, &[]);
    HostDonor { states: program.record.into_iter().map(|(k, v)| (k, v.expect("donor state reached"))).collect() }
}

#[test]
fn the_device_decoder_runs_every_program_and_action_as_the_cpu() {
    let dir = tiny_export("eval_device", 2);
    let decoder = Decoder::from_export(&dir).expect("decoder");
    let length = 12;
    let passages: Vec<Vec<u32>> = vec![(0..length).map(|t| (t * 7 % 11) as u32).collect(), (0..length).map(|t| (t * 3 % 11 + 1) as u32 % 11).collect()];
    let (libraries, selection) = coordinate(&decoder, length);
    let edit = (Arc::new(Array2::from_shape_fn((8, 2), |(i, j)| ((i + 3 * j) as f64 * 0.37).sin())), Arc::new(Array2::from_shape_fn((16, 2), |(i, j)| ((2 * i + j) as f64 * 0.21).cos() * 0.2)));
    let cases: Vec<Vec<Action>> = vec![
        vec![],
        vec![Action::Input { site: site_index(0, 5), change: InputChange::Scale { rows: Rows::One(6), cols: (3, 4), scale: 0.0 } }],
        vec![Action::Input { site: site_index(1, 3), change: InputChange::Scale { rows: Rows::All, cols: (2, 6), scale: 2.0 } }],
        vec![
            Action::Input { site: site_index(0, 1), change: InputChange::Mix { row: 7, alpha: 0.5 } },
            Action::Output { site: site_index(1, 4), change: OutputChange::Mix { row: 8, alpha: 1.0 } },
        ],
        vec![Action::Output { site: site_index(0, 0), change: OutputChange::Mix { row: 6, alpha: 0.25 } }],
        vec![Action::Output { site: site_index(1, 5), change: OutputChange::Add { left: edit.0.clone(), right: edit.1.clone() } }],
    ];
    let interface = [6usize, 9];
    for device in devices() {
        let resident = Resident::new(&device, &decoder, length).expect("resident");
        let units: Vec<Option<Units>> = libraries.iter().enumerate().map(|(k, l)| l.as_ref().map(|l| resident.units(k, &l.v, &l.u).expect("units"))).collect();
        let columns: Vec<usize> = libraries.iter().map(|l| l.as_ref().map_or(0, |l| l.v.nrows())).collect();
        // Clean runs of both programs on both passages, the donor states recorded on passage 1.
        let batch = resident.batch(&passages, &[0, 1], length, 0).expect("batch");
        let keys: Vec<(usize, usize, bool)> = cases.iter().flatten().filter_map(Action::donor_state).collect();
        let rule = GivenRows { selections: vec![&selection, &selection], columns: columns.clone() };
        let native_clean = resident.forward(&batch, &Run::clean(Maps::Native, 2, vec![keys.clone(), keys.clone()], Arithmetic::F64)).expect("native clean");
        let explained_clean = resident.forward(&batch, &Run::clean(Maps::Units { units: units.iter().map(Option::as_ref).collect(), rule: &rule }, 2, vec![keys.clone(), keys.clone()], Arithmetic::F64)).expect("explained clean");
        let mut native_kept = native_clean.kept.expect("kept");
        let mut explained_kept = explained_clean.kept.expect("kept");
        let native_donor: &Donor = &native_clean.recorded[1];
        let explained_donor: &Donor = &explained_clean.recorded[1];
        for (case, actions) in cases.iter().enumerate() {
            let host_native_donor = host_donor(&decoder, HostMaps::Native(&decoder), &passages[1], &keys);
            let mut donor_selection = selection.clone();
            let host_explained_donor = host_donor(&decoder, HostMaps::Units { decoder: &decoder, libraries: &libraries, selector: &mut donor_selection }, &passages[1], &keys);
            let mut native = Program::new(HostMaps::Native(&decoder), actions, Some(&host_native_donor));
            let expected_native = decoder.forward(&passages[0], &mut native, &interface);
            let mut chosen = selection.clone();
            let selector: &mut dyn Selector = &mut chosen;
            let mut explained = Program::new(HostMaps::Units { decoder: &decoder, libraries: &libraries, selector }, actions, Some(&host_explained_donor));
            let expected_explained = decoder.forward(&passages[0], &mut explained, &interface);
            let from = actions.iter().map(Action::first_row).min().unwrap_or(0);
            let first_layer = actions.iter().map(|a| a.site() / 6).min().unwrap_or(0);
            // From row and layer 0, and from the actions' first row and layer.
            for (start, layer) in [(0, 0), (from, first_layer)] {
                let batch = resident.batch(&passages, &[0], length, start).expect("batch");
                let one_rule = GivenRows { selections: vec![&selection], columns: columns.clone() };
                let clean_native: Vec<&Clean> = vec![&native_kept[0]];
                let clean_explained: Vec<&Clean> = vec![&explained_kept[0]];
                for (maps, donor, clean, expected) in [
                    (Maps::Native, native_donor, clean_native, &expected_native),
                    (Maps::Units { units: units.iter().map(Option::as_ref).collect(), rule: &one_rule }, explained_donor, clean_explained, &expected_explained),
                ] {
                    let run = Run { maps, actions: vec![actions], donors: vec![Some(donor)], interface: vec![&interface], record: vec![Vec::new()], from_layer: layer, clean: Some(clean), keep: false, arithmetic: Arithmetic::F64 };
                    let ran = resident.forward(&batch, &run).expect("forward");
                    let residual = device.download(&ran.residual).expect("download");
                    let want = expected.residual.slice(ndarray::s![start.., ..]).to_owned();
                    let worst = gap(&residual, &want);
                    assert!(worst < 1e-10, "{} case {case} from ({start}, {layer}): residual off by {worst}", device.name());
                    for (l, rows) in ran.interface[0].iter().enumerate() {
                        let got = device.download(rows).expect("download");
                        let worst = gap(&got, &expected.layers[l]);
                        assert!(worst < 1e-10, "{} case {case} from ({start}, {layer}): layer {l} interface off by {worst}", device.name());
                    }
                }
            }
        }
        // A clean run's units per row are what the rule chose.
        let per_row: Vec<f64> = (0..length).map(|r| selection.rows[r].len() as f64).collect();
        assert_eq!(explained_kept[0].units, per_row);
        native_kept.clear();
        explained_kept.clear();
    }
    std::fs::remove_dir_all(&dir).expect("remove the temporary export");
}
