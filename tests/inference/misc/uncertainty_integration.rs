

// `stateless_sas_inverse_link_is_rejected` was deleted: the type system now
// rejects `InverseLink::Standard(LinkFunction::Sas)` at compile time
// (`InverseLink::Standard` carries `StandardLink`, which has no `Sas`
// variant), so the runtime check it exercised is unreachable by
// construction.

