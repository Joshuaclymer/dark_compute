import math

from slowdown_model_paramaters import TakeoverRiskParameters


class TakeoverRiskModel:
    # Time anchor points in years - change these to adjust the interpolation points
    ANCHOR_T1 = 1/12  # 1 month (first anchor)
    ANCHOR_T2 = 1.0   # 1 year (second anchor)
    ANCHOR_T3 = 10.0  # 10 years (third anchor)

    # Alignment tax anchor points (proportion of compute spent on alignment, 0-1)
    ALIGNMENT_TAX_T1 = 0.01  # 1% of compute on alignment
    ALIGNMENT_TAX_T2 = 0.10  # 10% of compute on alignment
    ALIGNMENT_TAX_T3 = 1.00  # 100% of compute on alignment

    @staticmethod
    def compute_safety_speedup(capability_speedup: float, multiplier: float) -> float:
        """
        Compute safety research speedup from capability research speedup.

        Safety research speedup = capability_speedup * multiplier

        For example, if capability speedup is 4x and multiplier is 0.5,
        safety speedup is 4 * 0.5 = 2x.

        Args:
            capability_speedup: The AI capability research speedup factor
            multiplier: The multiplier relating safety to capability speedup

        Returns:
            Safety research speedup factor
        """
        if capability_speedup <= 0:
            return 1.0
        return capability_speedup * multiplier

    def _interpolate_probability(self, duration_years: float, p_t1: float, p_t2: float, p_t3: float) -> float:
        """
        Interpolate probability using log-linear interpolation on duration.

        Uses the logit transform to ensure probabilities stay bounded (0, 1).
        Interpolates in logit space for smoother curves, with log-time for x-axis.

        Args:
            duration_years: The handoff duration in years
            p_t1: Probability at ANCHOR_T1 (default: 1 month)
            p_t2: Probability at ANCHOR_T2 (default: 1 year)
            p_t3: Probability at ANCHOR_T3 (default: 10 years)

        Returns:
            Interpolated probability bounded between 0 and 1
        """
        # Time anchor points in years
        t1 = self.ANCHOR_T1
        t2 = self.ANCHOR_T2
        t3 = self.ANCHOR_T3

        # Clamp duration to valid range (minimum is t1 for log interpolation)
        duration_years = max(t1, min(duration_years, t3 * 10))

        # Convert probabilities to logit space for interpolation
        def logit(p):
            p = max(1e-10, min(1 - 1e-10, p))  # Clamp to avoid log(0)
            return math.log(p / (1 - p))

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        logit_p1 = logit(p_t1)
        logit_p2 = logit(p_t2)
        logit_p3 = logit(p_t3)

        # Use log-time for interpolation (log-linear in time)
        log_t1 = math.log(t1)
        log_t2 = math.log(t2)
        log_t3 = math.log(t3)
        log_duration = math.log(duration_years)

        # Piecewise linear interpolation in log-time, logit-probability space
        if log_duration <= log_t2:
            # Interpolate between 1 month and 1 year
            alpha = (log_duration - log_t1) / (log_t2 - log_t1)
            logit_p = logit_p1 + alpha * (logit_p2 - logit_p1)
        else:
            # Interpolate between 1 year and 10 years
            alpha = (log_duration - log_t2) / (log_t3 - log_t2)
            logit_p = logit_p2 + alpha * (logit_p3 - logit_p2)

        return sigmoid(logit_p)

    def p_misalignment_at_handoff(self, adjusted_alignment_research_time: float, params: TakeoverRiskParameters) -> float:
        """
        Estimate the probability of misalignment at handoff given the adjusted alignment research time.

        P(misalignment at handoff) is a function of alignment research time between present day
        and handoff, adjusted for Research Speedup, Research Relevance, and Slowdown Effort.
        See slowdown_model.md for details.

        The function extrapolates from three anchor points:
        - P(misalignment at handoff) if adjusted research time = 1 month
        - P(misalignment at handoff) if adjusted research time = 1 year
        - P(misalignment at handoff) if adjusted research time = 10 years

        Args:
            adjusted_alignment_research_time (float): Alignment research time adjusted for
                speedup, relevance, and slowdown effort. For example, if there's a 2x AI R&D
                speedup during 1 year of calendar time with exponent 0.5, the adjusted time
                would be sqrt(2) * 1 = 1.41 years.

        Returns:
            float: Estimated probability of misalignment at handoff.
        """
        return self._interpolate_probability(
            adjusted_alignment_research_time,
            params.p_misalignment_at_handoff_t1,
            params.p_misalignment_at_handoff_t2,
            params.p_misalignment_at_handoff_t3
        )

    def p_AI_takeover(self, research_speed_adjusted_handoff_duration: float, params: TakeoverRiskParameters) -> float:
        """
        Alias for p_misalignment_at_handoff for backward compatibility.

        Note: This function is deprecated. Use p_misalignment_at_handoff instead.
        The naming "p_AI_takeover" was misleading as this estimates P(misalignment at handoff),
        not the full P(AI takeover) which also includes P(misalignment after handoff).

        Args:
            research_speed_adjusted_handoff_duration (float): The adjusted alignment research time.

        Returns:
            float: Estimated probability of misalignment at handoff.
        """
        return self.p_misalignment_at_handoff(research_speed_adjusted_handoff_duration, params)

    def p_human_power_grabs(self, takeoff_trajectory: dict, params: TakeoverRiskParameters) -> float:
        """
        Estimate the probability of human power grabs given a takeoff trajectory.

        The risk of human power grabs depends on how much time society has to react
        to powerful AI. A slower takeoff poses lower risk. This function uses the
        time between Superhuman AI Researcher (SAR) and ASI as the key duration,
        since this is when the world will most acutely recognize the possibility
        of takeover.

        Args:
            takeoff_trajectory (dict): Response from TakeoffModel.predict_trajectory(),
                containing milestones_median (or milestones for deterministic version)
                with SAR and ASI timing information.
        Returns:
            float: Estimated probability of human power grabs
        """
        # Extract milestones from trajectory (handle both stochastic and deterministic formats)
        milestones = takeoff_trajectory.get('milestones_median') or takeoff_trajectory.get('milestones', {})

        # Get SAR and ASI times
        # Note: SAR milestone uses the full name 'SAR-level-experiment-selection-skill'
        sar_info = milestones.get('SAR-level-experiment-selection-skill', {}) or milestones.get('SAR', {})
        asi_info = milestones.get('ASI', {})

        sar_time = sar_info.get('time') if isinstance(sar_info, dict) else None
        asi_time = asi_info.get('time') if isinstance(asi_info, dict) else None

        # If we can't determine the duration, return a default high-risk estimate
        if sar_time is None or asi_time is None:
            # Default to 1 month duration (high risk) if milestones unavailable
            sar_to_asi_duration = self.ANCHOR_T1
        else:
            sar_to_asi_duration = asi_time - sar_time
            # Ensure non-negative duration
            sar_to_asi_duration = max(self.ANCHOR_T1 / 10, sar_to_asi_duration)

        return self._interpolate_probability(
            sar_to_asi_duration,
            params.p_human_power_grabs_t1,
            params.p_human_power_grabs_t2,
            params.p_human_power_grabs_t3
        )

    def _compute_alignment_tax_for_target_probability(
        self,
        target_probability: float,
        params: TakeoverRiskParameters
    ) -> float:
        """
        Compute the alignment tax required to achieve a target probability of misalignment.

        This inverts the p_misalignment_at_handoff function to find:
        "What adjusted alignment research time produces this probability?"

        Then computes the alignment tax as:
        tax = adjusted_alignment_time / adjusted_capability_time

        For the handoff period, we assume capability research time equals calendar time
        (i.e., normalized to 1), so the tax equals the adjusted alignment research time.

        Args:
            target_probability: The target P(misalignment at handoff)
            params: TakeoverRiskParameters with anchor points

        Returns:
            The alignment tax (ratio) needed to achieve that probability
        """
        # We need to invert the interpolation to find the duration that gives target_probability
        # The interpolation maps duration -> probability via logit space
        # We need to find duration given probability

        def logit(p):
            p = max(1e-10, min(1 - 1e-10, p))
            return math.log(p / (1 - p))

        t1 = self.ANCHOR_T1
        t2 = self.ANCHOR_T2
        t3 = self.ANCHOR_T3

        p_t1 = params.p_misalignment_at_handoff_t1
        p_t2 = params.p_misalignment_at_handoff_t2
        p_t3 = params.p_misalignment_at_handoff_t3

        logit_p1 = logit(p_t1)
        logit_p2 = logit(p_t2)
        logit_p3 = logit(p_t3)
        logit_target = logit(target_probability)

        log_t1 = math.log(t1)
        log_t2 = math.log(t2)
        log_t3 = math.log(t3)

        # Determine which segment we're in and invert
        if logit_target >= logit_p2:
            # We're in the t1-t2 segment (higher probability = shorter duration)
            if logit_p1 == logit_p2:
                log_duration = log_t1
            else:
                alpha = (logit_target - logit_p1) / (logit_p2 - logit_p1)
                log_duration = log_t1 + alpha * (log_t2 - log_t1)
        else:
            # We're in the t2-t3 segment
            if logit_p2 == logit_p3:
                log_duration = log_t2
            else:
                alpha = (logit_target - logit_p2) / (logit_p3 - logit_p2)
                log_duration = log_t2 + alpha * (log_t3 - log_t2)

        duration = math.exp(log_duration)
        # The alignment tax is the adjusted alignment time needed
        # (assuming capability time is normalized to 1)
        return duration

    def _interpolate_probability_alignment_tax(
        self,
        alignment_tax: float,
        p_t1: float,
        p_t2: float,
        p_t3: float
    ) -> float:
        """
        Interpolate probability using log-linear interpolation on alignment tax.

        Similar to _interpolate_probability but uses alignment tax anchor points (0.01, 0.10, 1.00)
        instead of time anchor points.

        Args:
            alignment_tax: The proportion of compute spent on alignment (0 to 1)
            p_t1: Probability at ALIGNMENT_TAX_T1 (1% of compute)
            p_t2: Probability at ALIGNMENT_TAX_T2 (10% of compute)
            p_t3: Probability at ALIGNMENT_TAX_T3 (100% of compute)

        Returns:
            Interpolated probability bounded between 0 and 1
        """
        # Tax anchor points
        t1 = self.ALIGNMENT_TAX_T1  # 0.01
        t2 = self.ALIGNMENT_TAX_T2  # 0.10
        t3 = self.ALIGNMENT_TAX_T3  # 1.00

        # Clamp alignment tax to valid range
        alignment_tax = max(t1 / 10, min(alignment_tax, 1.0))

        # Convert probabilities to logit space for interpolation
        def logit(p):
            p = max(1e-10, min(1 - 1e-10, p))  # Clamp to avoid log(0)
            return math.log(p / (1 - p))

        def sigmoid(x):
            return 1 / (1 + math.exp(-x))

        logit_p1 = logit(p_t1)
        logit_p2 = logit(p_t2)
        logit_p3 = logit(p_t3)

        # Use log-tax for interpolation (log-linear in alignment tax)
        log_t1 = math.log(t1)
        log_t2 = math.log(t2)
        log_t3 = math.log(t3)
        log_tax = math.log(alignment_tax)

        # Piecewise linear interpolation in log-tax, logit-probability space
        if log_tax <= log_t2:
            # Interpolate between 1% and 10%
            alpha = (log_tax - log_t1) / (log_t2 - log_t1)
            logit_p = logit_p1 + alpha * (logit_p2 - logit_p1)
        else:
            # Interpolate between 10% and 100%
            alpha = (log_tax - log_t2) / (log_t3 - log_t2)
            logit_p = logit_p2 + alpha * (logit_p3 - logit_p2)

        return sigmoid(logit_p)

    def p_misalignment_after_handoff(
        self,
        alignment_tax_after_handoff: float,
        params: TakeoverRiskParameters
    ) -> float:
        """
        Estimate the probability of misalignment after handoff given the alignment tax paid.

        The alignment tax is defined as the proportion of compute spent on alignment
        after handoff (a value between 0 and 1). This function uses a separate curve
        from the pre-handoff misalignment curve, with its own anchor points.

        The curve maps:
        - 1% compute on alignment → p_misalignment_after_handoff_t1
        - 10% compute on alignment → p_misalignment_after_handoff_t2
        - 100% compute on alignment → p_misalignment_after_handoff_t3

        Args:
            alignment_tax_after_handoff: The proportion of compute devoted to alignment
                after handoff (0 to 1). For example, 0.1 means 10% of compute is spent
                on alignment research.
            params: TakeoverRiskParameters with anchor points

        Returns:
            float: Estimated probability of misalignment after handoff
        """
        return self._interpolate_probability_alignment_tax(
            alignment_tax_after_handoff,
            params.p_misalignment_after_handoff_t1,
            params.p_misalignment_after_handoff_t2,
            params.p_misalignment_after_handoff_t3
        )

    def p_AI_takeover_full(
        self,
        adjusted_alignment_research_time_at_handoff: float,
        alignment_tax_after_handoff: float,
        params: TakeoverRiskParameters
    ) -> float:
        """
        Compute full P(AI takeover) combining misalignment at and after handoff.

        Per slowdown_model.md:
        P(takeover) = 1 - [1 - P(misalignment at handoff)] * [1 - P(misalignment after handoff)]

        Args:
            adjusted_alignment_research_time_at_handoff: Alignment research time up to handoff,
                adjusted for speedup, relevance, and slowdown effort
            alignment_tax_after_handoff: The ratio of adjusted alignment research time to
                adjusted capability research time for the post-handoff period
            params: TakeoverRiskParameters with probability anchor points

        Returns:
            Combined probability of AI takeover from misalignment
        """
        p_at_handoff = self.p_misalignment_at_handoff(
            adjusted_alignment_research_time_at_handoff, params
        )
        p_after_handoff = self.p_misalignment_after_handoff(
            alignment_tax_after_handoff, params
        )
        return 1 - (1 - p_at_handoff) * (1 - p_after_handoff)

    def p_domestic_takeover(
        self,
        takeoff_trajectory: dict,
        adjusted_alignment_research_time: float,
        params: TakeoverRiskParameters,
        alignment_tax_after_handoff: float = None
    ) -> float:
        """
        Compute the combined probability of catastrophe from AI takeover and human power grabs.

        Per slowdown_model.md, P(AI takeover) combines:
        - P(misalignment at handoff): based on adjusted alignment research time up to handoff
        - P(misalignment after handoff): based on alignment tax paid after handoff

        Then P(catastrophe) combines P(AI takeover) with P(human power grabs):
        P(catastrophe) = 1 - (1 - P(AI takeover)) * (1 - P(human power grabs))

        Args:
            takeoff_trajectory: Response from TakeoffModel.predict_trajectory()
            adjusted_alignment_research_time: Alignment research time adjusted for speedup,
                relevance, and slowdown effort (for the period up to handoff)
            params: TakeoverRiskParameters with probability anchor points
            alignment_tax_after_handoff: Optional. The ratio of adjusted alignment research time
                to adjusted capability research time for the post-handoff period.
                If None, P(misalignment after handoff) is assumed to be 0.

        Returns:
            Combined probability of catastrophe
        """
        # Compute P(misalignment at handoff)
        p_misalignment_at = self.p_misalignment_at_handoff(adjusted_alignment_research_time, params)

        # Compute P(misalignment after handoff) if alignment tax is provided
        if alignment_tax_after_handoff is not None and alignment_tax_after_handoff > 0:
            p_misalignment_after = self.p_misalignment_after_handoff(
                alignment_tax_after_handoff, params
            )
        else:
            # If no post-handoff data, assume no additional risk from post-handoff period
            p_misalignment_after = 0.0

        # Combine into P(AI takeover)
        p_ai_takeover = 1 - (1 - p_misalignment_at) * (1 - p_misalignment_after)

        # Compute P(human power grabs)
        p_power_grabs = self.p_human_power_grabs(takeoff_trajectory, params)

        # Final P(catastrophe)
        return 1 - (1 - p_ai_takeover) * (1 - p_power_grabs)