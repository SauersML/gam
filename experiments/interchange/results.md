# Qwen Calendar Interchange

## Headline

- Interchange accuracy: **0.0476** (2/42)
- Mean realized KL(clean base || patched): **0.006991 nats**
- Mean predicted-label restricted log-prob lift: **0.1167 nats**
- Mean predicted-label logit margin: **-3.3452**

## Protocol

- Model: `qwen3-8b`
- Feature: `weekday` with 7 periodic labels
- Patched block: module index `17` (hidden-state L18)
- Atom fit: one periodic harmonic decoder, harmonics `2`, ridge `0.001`
- Fit activation EV on train templates: `0.028093`
- Clean restricted accuracy: train `1.0000`, eval `0.6667`
- Interventions: `42` held-out base/source swaps
- Match tolerance: predicted label logit within `0.0` of restricted top-1

The fitted atom maps known calendar phase to the residual-stream coordinate at the patched block. For each held-out intervention, the source token's phase is decoded through that atom and delta-written into the base token residual while preserving the base prompt. The predicted behavior is the source phase's known next-calendar label.

## Sample Interventions

| base | source | predicted | realized | match | KL | predicted margin |
|---|---|---|---|---:|---:|---:|
| Wednesday | Thursday | Friday | Thursday | false | 0.014011 | -3.2500 |
| Friday | Monday | Tuesday | Saturday | false | 0.008114 | -5.7500 |
| Friday | Tuesday | Wednesday | Saturday | false | 0.006087 | -5.6250 |
| Friday | Wednesday | Thursday | Saturday | false | 0.012507 | -6.1250 |
| Saturday | Monday | Tuesday | Sunday | false | 0.023304 | -4.0000 |
| Saturday | Tuesday | Wednesday | Sunday | false | 0.019771 | -4.3750 |
| Saturday | Thursday | Friday | Sunday | false | 0.025456 | -5.8125 |
| Sunday | Monday | Tuesday | Monday | false | 0.010779 | -2.6250 |
| Sunday | Friday | Saturday | Monday | false | 0.010776 | -5.7500 |
| Sunday | Saturday | Sunday | Monday | false | 0.005224 | -6.1250 |
| Monday | Thursday | Friday | Tuesday | false | 0.002008 | -5.7500 |
| Monday | Friday | Saturday | Tuesday | false | 0.001071 | -7.2500 |

## Artifacts

- `numbers.json` contains every clean prompt and intervention record.
