#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum RowScaleMode {
    Signed,
    SqrtPositive,
    AbsWithSign,
}
