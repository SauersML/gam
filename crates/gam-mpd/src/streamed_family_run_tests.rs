use super::*;
use crate::acceptance::Change;

#[test]
fn hidden_teacher_prefix_and_tiled_head_preserve_native_values_and_edits() {
    let dir=crate::test_support::tiny_export("streamed_family_hidden",2);
    let imported=crate::import::import_language_model(&dir,1,12).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    let head=Head::of(&imported.program).unwrap();
    let prefix=head.prefix(&imported.program).unwrap();
    assert_eq!(prefix.output,head.hidden);
    assert_eq!(prefix.nodes,imported.program.nodes[..=head.hidden]);
    let width=imported.program.node_interface(head.hidden).unwrap().width();
    let changes=[vec![],vec![Edit {node:head.hidden,rows:Some(vec![0,3,3]),columns:0..width,change:Change::Scale(0.5)}]];
    for edits in changes {
        let execute=|p:&OperatorProgram|p.execute_edited(&imported.contract.family,|node,value,_| {
            apply(&edits.iter().filter(|e|e.node==node).collect::<Vec<_>>(),value);Ok(())
        }).unwrap();
        let full=execute(&imported.program);
        let hidden=execute(&prefix);
        assert_eq!(hidden.values[prefix.output],full.values[head.hidden]);
        let expected=&full.values[imported.program.output];
        for tile in [1,5,12] {
            for start in (0..12).step_by(tile) {
                let end=(start+tile).min(12);
                let actual=head.logits(&imported.program,hidden.values[prefix.output].slice(s![start..end,..])).unwrap();
                assert_eq!(actual.dim(),(end-start,head.classes));
                for (&a,&b) in actual.iter().zip(expected.slice(s![start..end,..]).iter()) {
                    assert!((a-b).abs()<=1e-12,"tiled native head differs: {a} versus {b}");
                }
            }
        }
    }
}

#[test]
fn numeric_budgets_count_hidden_teachers_and_all_four_logits_tiles() {
    let head=Head {hidden:4,operator:0,transposed:false,classes:151936,width:1024};
    let retained=81*1024*1024*8;
    let per_row=(4*151936+2*1024)*8;
    let budget=StreamedBudget {teacher_bytes:retained,tile_bytes:7*per_row,intermediate_bytes:1};
    assert_eq!(layout(&head,1024,80,budget).unwrap(),(retained,7));
    assert!(layout(&head,1024,80,StreamedBudget {teacher_bytes:retained-1,..budget}).is_err());
    assert!(layout(&head,1024,80,StreamedBudget {tile_bytes:per_row-1,..budget}).is_err());
    assert!(layout(&head,usize::MAX,80,budget).is_err());
    // 81 full vocabulary arrays would be more than 90GiB; cache actual hidden states.
    assert!(retained<1024*1024*1024);
    assert!(81_u64*1024*151936*8>90*1024*1024*1024);
}

#[test]
fn native_plan_rejects_non_dense_or_multi_term_terminal_head() {
    let dir=crate::test_support::tiny_export("streamed_family_head_guard",1);
    let imported=crate::import::import_language_model(&dir,1,4).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    let mut program=imported.program;
    let head=Head::of(&program).unwrap();
    let mut edit=Edit {node:head.hidden,rows:None,columns:0..1,change:Change::Scale(0.0)};
    assert!(head.allows_edits(&[edit.clone()]).is_ok());
    edit.node=program.output;
    assert!(head.allows_edits(&[edit]).is_err());
    program.nodes.push(Node::Affine {terms:vec![(head.hidden,head.operator),(head.hidden,head.operator)],bias:None});
    program.output=program.nodes.len()-1;
    assert!(Head::of(&program).is_err());
}

#[test]
fn values_prefix_preserves_ordered_partial_native_edits_and_source() {
    let dir=crate::test_support::tiny_export("streamed_native_values",2);
    let imported=crate::import::import_language_model(&dir,1,6).unwrap();
    std::fs::remove_dir_all(dir).unwrap();
    let head=Head::of(&imported.program).unwrap();
    let prefix=head.prefix(&imported.program).unwrap();
    let edits=vec![
        Edit {node:head.hidden,rows:Some(vec![1,1,4]),columns:0..1,change:Change::Scale(0.3)},
        Edit {node:head.hidden,rows:Some(vec![1,4]),columns:0..1,change:Change::Add(0.125)},
    ];
    let device=Device::host();
    let compiled=crate::device_program::DeviceProgram::compile_values(&device,&prefix).unwrap();
    let clean=prefix.execute(&imported.contract.family,false).unwrap().values[prefix.output].clone();
    for changes in [edits,vec![]] {
        let expected=prefix.execute_edited(&imported.contract.family,|node,value,_| {
            apply(&changes.iter().filter(|e|e.node==node).collect::<Vec<_>>(),value);Ok(())
        }).unwrap();
        let trace=compiled.forward_edited(&imported.contract.family,std::collections::BTreeMap::new(),|_,_|Ok(()),|node,trace| {
            let changes=changes.iter().filter(|e|e.node==node).collect::<Vec<_>>();
            if changes.is_empty(){return Ok(None);}
            let mut value=device.download(trace.value(node)?).map_err(|e|e.to_string())?;
            apply(&changes,&mut value);
            Ok(Some(device.upload(value.view()).map_err(|e|e.to_string())?))
        }).unwrap();
        let actual=device.download(trace.value(prefix.output).unwrap()).unwrap();
        for (a,b) in actual.iter().zip(expected.values[prefix.output].iter()) {assert!((a-b).abs()<1e-12);}
    }
    assert_eq!(prefix.execute(&imported.contract.family,false).unwrap().values[prefix.output],clean);
}

#[test]
fn native_head_device_orientation_preserves_sub_f32_values() {
    let device=Device::host();
    let hidden=ndarray::arr2(&[[1.0+2f64.powi(-40),-2.0],[0.2,3.0]]);
    let matrix=ndarray::arr2(&[[0.5,1.0+2f64.powi(-35)],[-0.3,0.7],[2.0,-1.0]]);
    let expected=hidden.dot(&matrix.t());
    for transposed in [false,true] {
        let values=if transposed {matrix.t().to_owned()}else{matrix.clone()};
        let tensor=device.upload(values.view()).unwrap();
        let head=Head {hidden:0,operator:0,transposed,classes:3,width:2};
        let actual=head.logits_device(&device,&tensor,hidden.view()).unwrap();
        for (a,b) in actual.iter().zip(expected.iter()) {assert!((a-b).abs()<1e-14);}
        assert_eq!(device.download(&tensor).unwrap(),values);
        assert!(head.logits_device(&device,&tensor,hidden.slice(s![..,..1])).is_err());
    }
}
