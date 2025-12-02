# Slowdown Model

## Phases of agreements

There are three main phases of the AI Slowdown Agreement:
1. Cease training 
2. Cap and align
3. Handoff

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

The alignment tax after handoff is defined as the **proportion of compute spent on alignment research** (a value between 0 and 1). For example, an alignment tax of 0.10 means 10% of compute is devoted to alignment research after handoff.

This is a user-specified input parameter (`alignment_tax_after_handoff`) rather than a computed value.

### Mapping Tax to Risk

P(misalignment after handoff) is a function of the alignment tax. The function is obtained by extrapolating between three anchor points:

- P(misalignment after handoff) if alignment tax = 1% of compute
- P(misalignment after handoff) if alignment tax = 10% of compute
- P(misalignment after handoff) if alignment tax = 100% of compute

The interpolation uses log-linear interpolation in logit-probability space (same technique as the other probability curves).

### Parameters

- `alignment_tax_after_handoff`: The proportion of compute spent on alignment (0 to 1)
- `p_misalignment_after_handoff_t1`: P(misalignment) at 1% alignment tax
- `p_misalignment_after_handoff_t2`: P(misalignment) at 10% alignment tax
- `p_misalignment_after_handoff_t3`: P(misalignment) at 100% alignment tax

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


### Estimating Optimal Proxy project compute trajectory

The way we compute the optimal proxy project compute trajectory is to first of all turn the problem into a myopic one: What should the compute allowance be for the proxy project just over the next year? And then when the next year comes along, we can ask what the compute allowance should be for the year after that. Etc.

At each year, we can make a myopia assumption, which is that, at the end of the year, the agreement will end. we choose the optimal compute allowance for the proxy project assuming the agreement will end at the end of the year. I'm not claiming this is the optimal way to do it globally. We're just going to make a myopic assumption for computational tractability.

Okay, so how do we determine what the optimal compute allowance is for a given year? This is an optimization problem. There are many ways to solve optimization problems. A simple way is discrete search. Let's start with that. First come up with candidates of compute allowances, Then for each candidate, compute the following:
- The domestic takeover risk assuming (1) That compute allowance is held for the entire year and (2) The agreement ends at the end of the year - so the US project starts using all of its compute for AI R&D and its compute starts growing at the standard rate again.
- The risk that *china 
- 

We're interested in the instantaneous value of P(takeover) -- the value if all compute caps and restrictions were suddenly lifted -- at each point during the agreement.

To do this, we need to determine a takeoff curve for the US if the agreement suddenly ended. The expensive way to do this would be to compute the takeoff curve using the takeoff model. But there's potentially a shortcut that's much faster:
- Obtain the takeoff curve for the no-slowdown counterfactual.
- Identify the current Y value (AI R&D speedup) when the slowdown suddenly lifts
- Identify the part of the no-slowdown counterfactual curve with the same Y value. 
- Use this part of the no-slowdown counterfactual curve to complete the rest of the trajectory. 


## Proxy project
The U.S. proxy project is supposed to be a proxy for the capabilities that the PRC or other uncooperative actors might achieve covertly.

The amount of compute the proxy project has is given by the parameters in the ProxyProject class:

```
class ProxyProjectParameters:
    """Parameters for the US slowdown trajectory (compute cap based on PRC covert compute)."""
    compute_cap_as_percentile_of_PRC_operational_covert_compute: float = 0.7 #70th percentile
    determine_optimal_proxy_project_compute_based_on_risk_curves : bool = False
    frequency_cap_is_updated_in_years: float = 1.0
```

When "determine_optimal_proxy_project_compute_based_on_risk_curves : bool = True," please determine the proxy project compute cap as follows:
- 

## Capability cap
The capability cap is a cap on the AR&D speedup of the US project during the AI slowdown.

**Before the agreement starts:** There is no capability cap. The US (slowdown) trajectory should be identical to the US (no slowdown) trajectory.

**After the agreement starts but before the capability cap can be enforced with evaluations** (years_after_agreement_start_when_evaluation_based_capability_cap_is_implemented): The capability cap is set to whatever capabilities the proxy project can achieve.

After the capability cap can be enforced with evaluations, the capability cap is set to the maximum of the following:
- Some fixed AI R&D speedup (capability_cap_ai_randd_speedup). By default, this is set at the point where developers cannot effectively oversee AI alignment work (hand-off).
- The capabilities achieved by the 'proxy project.'

## US project compute spending during the AI slowdown

Before the AI slowdown, the US spends compute on AI experiments equal to the compute spent by the US if there is no slowdown.

After the slowdown first begins, there's a period "years_after_agreement_start_when_evaluation_based_capability_cap_is_implemented" during which the largest US project uses compute for experiments equal to the 

The US project expends compute on experiments so as to avoid exceeding the capability cap. The way this works is sort of like a thermostat. The capability cap is like the
temperature setting of a thermostat. The developer starts out using compute for experiments equal to the compute available to the proxy project. Then the developer increases compute used for experiments to reach the capability cap (temperature). But here's the key: the developer doesn't want to overshoot the cap. So as
developers approach the capability cap, developers increase experiment compute more gradually to
 avoid overshooting it.


 Currently, we binary search for caps on the progress,
  such that the IR&D speedup remains below the capability cap.