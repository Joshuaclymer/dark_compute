# Slowdown Model

## Modeling AI Takeover Risk

### Overview

This framework decomposes takeover risk into two components:

```
P(takeover) = 1 − [1 − P(misalignment at handoff)] × [1 − P(misalignment after handoff)]
```

### Key Definitions

- **Present day:** 2026
- **Automated coder:** The capability threshold marking the start of the "handoff window"
- **Handoff:** When AI systems achieve a 20x AI R&D speedup; after this point, alignment research is fully automated without effective human oversight

---

## Estimating P(Misalignment at Handoff)

P(misalignment at handoff) is a function of alignment research time between present day and handoff, adjusted for three factors: Research Speedup, Research Relevance, and Slowdown Effort.

To obtain this function, we'll extrapolate from three anchor points obtained by surveying experts:

- P(misalignment at handoff) if adjusted research time = 1 month
- P(misalignment at handoff) if adjusted research time = 1 year
- P(misalignment at handoff) if adjusted research time = 10 years

### 1. Research Speedup Adjustment

Calculate the alignment speedup by raising the AI R&D speedup to an exponent (e.g., 0.5). Integrate this alignment speedup curve over time to obtain speed-up-adjusted alignment research time.

### 2. Research Relevance Adjustment

Divide the pre-handoff period into two windows:

- **Pre-handoff window:** Present day → automated coder
- **Handoff window:** Automated coder → handoff

Apply a `research_relevance_of_pre_handoff_discount` parameter (a value between 0 and 1, e.g., 0.1) to the speedup-adjusted alignment research occurring during the pre-handoff window. The handoff window receives no discount (multiplier of 1).

### 3. Slowdown Effort Adjustment

Multiply any adjusted alignment research time occurring during an AI slowdown by `increase_in_alignment_research_effort_during_slowdown`.

---

## Estimating P(Misalignment After Handoff)

### The Alignment Tax

The alignment tax paid is the ratio of research labor expended on alignment versus capabilities.

```
Alignment tax paid during handoff = [speedup-and-effort-adjusted alignment research time during handoff window] ÷ [speedup-adjusted capabilities research time during handoff window]
```

### Mapping Tax to Risk

Using the model above, construct a mapping from "alignment tax paid during handoff" to "P(misalignment at handoff)."

Then introduce `alignment_tax_after_handoff_relative_to_during_handoff`—a constant representing how much higher the required alignment tax is after handoff. For example, if this constant is 2, the alignment tax required after handoff is always 2× higher than during handoff.

Use this constant to create a corresponding function mapping "alignment tax paid after handoff" to "P(misalignment after handoff)."

### Computing the Post-Handoff Estimate

1. Compute speedup-and-effort-adjusted alignment research time after handoff (using the same adjustment process as above)
2. Compute speedup-adjusted capabilities research time after handoff
3. Take the ratio to get the alignment tax paid after handoff
4. Plug this value into the "alignment tax paid after handoff" → "P(misalignment after handoff)" function

---

## Modeling Human Power Grab Risk Based on Takeoff Trajectory

### Overview

The risk of a human power grab depends on how much time society has to react to powerful AI. In general, a slower takeoff poses lower risk of human power grabs.

### Which Part of the Takeoff Curve Matters?

Not all segments of the takeoff trajectory are equally important. Slowing the transition from 1× to 2× AI R&D speedup would likely have minimal impact compared to slowing the transition between superhuman coder and superintelligence.

For simplicity, this model considers only the time between superhuman AI researcher and superintelligence, ignoring earlier milestones since this is the period when most of the wakeup to takeover-capable AI is likely to happen.

### Estimating P(Human Power Grab)

P(human power grab) is a function of the elapsed time between superhuman AI researcher (SAR) and superintelligence (ASI). The function is obtained by extrapolating between three anchor points, obtained by surveying experts:

- P(human power grab) if time between SAR and ASI is 1 month
- P(human power grab) if time between SAR and ASI is 1 year
- P(human power grab) if time between SAR and ASI is 10 years