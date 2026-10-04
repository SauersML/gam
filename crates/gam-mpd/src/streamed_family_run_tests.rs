use super::*;
use crate::acceptance::Change;

#[test]
fn hidden_teacher_prefix_and_tiled_head_preserve_native_values_and_edits() {
    let dir=crate::explanation_tests::tiny_export("streamed_family_hidden",2);
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
    let dir=crate::explanation_tests::tiny_export("streamed_family_head_guard",1);
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
