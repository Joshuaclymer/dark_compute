class ProxyProject:
    """
    Computes the capabilities trajectory for a proxy project.

    The proxy project represents what the PRC or other uncooperative actors might
    achieve covertly. Its compute is capped based on a percentile of the PRC
    operational covert compute distribution.

    The compute cap is updated at a specified frequency (creating step changes),
    and the capabilities trajectory is computed by running the takeoff model with
    this capped compute.
    """

    def __init__(self, params: Optional[ProxyProjectParameters] = None):
        """
        Initialize the ProxyProject.

        Args:
            params: ProxyProjectParameters containing:
                - compute_cap_as_percentile_of_PRC_operational_covert_compute: float (0-1)
                - frequency_cap_is_updated_in_years: float
        """
        self.params = params if params is not None else ProxyProjectParameters()
        self._compute_cap: Optional[Dict[str, Any]] = None
        self._trajectory: Optional[Dict[str, Any]] = None

    def compute_compute_cap(
        self,
        years: List[float],
        covert_compute_percentiles: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """
        Compute the compute cap for the proxy project over time.

        The compute cap is determined by taking a given percentile of the PRC operational
        covert compute distribution. The cap is updated at a specified frequency,
        creating a discrete/step-change curve.

        Args:
            years: List of years for the time series
            covert_compute_percentiles: Dict with keys like 'p10', 'p25', 'p50', 'p75', 'p90'
                containing lists of compute values at each percentile for each year

        Returns:
            Dictionary with:
                - 'years': List of years
                - 'compute': List of compute cap values
        """
        if not years or not covert_compute_percentiles:
            self._compute_cap = {'years': [], 'compute': []}
            return self._compute_cap

        percentile = self.params.compute_cap_as_percentile_of_PRC_operational_covert_compute
        update_frequency = self.params.frequency_cap_is_updated_in_years

        # Determine which percentile key to use based on the percentile value
        percentile_keys = {
            0.10: 'p10',
            0.25: 'p25',
            0.50: 'p50',
            0.75: 'p75',
            0.90: 'p90'
        }

        # Find the closest available percentile
        available_percentiles = sorted(percentile_keys.keys())
        closest_percentile = min(available_percentiles, key=lambda x: abs(x - percentile))
        percentile_key = percentile_keys[closest_percentile]

        # Get the covert compute values at the selected percentile
        if percentile_key not in covert_compute_percentiles:
            # Fallback to median if the exact percentile isn't available
            percentile_key = 'p50' if 'p50' in covert_compute_percentiles else 'median'

        if percentile_key not in covert_compute_percentiles:
            self._compute_cap = {'years': [], 'compute': []}
            return self._compute_cap

        covert_compute_at_percentile = covert_compute_percentiles[percentile_key]

        # Build the step-change curve
        cap_values = []
        current_cap = None
        last_update_year = None

        for i, year in enumerate(years):
            if last_update_year is None or (year - last_update_year) >= update_frequency:
                # Update the cap
                current_cap = covert_compute_at_percentile[i]
                last_update_year = year

            cap_values.append(current_cap if current_cap is not None else 0)

        self._compute_cap = {
            'years': list(years),
            'compute': cap_values
        }
        return self._compute_cap

    def compute_trajectory(
        self,
        covert_compute_data: Optional[Dict[str, Any]],
        takeoff_model: Any = None,
        human_labor: Optional[List[float]] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Compute the full capabilities trajectory for the proxy project.

        This runs the takeoff model with the compute-capped inputs to get
        the AI R&D speedup trajectory.

        Args:
            covert_compute_data: Dict with 'years' and 'operational_black_project' containing
                percentile data (p25, median/p50, p75)
            takeoff_model: TakeoffModel instance for computing trajectories
            human_labor: Optional list of human labor values. If None, uses default.

        Returns:
            Dictionary with trajectory data including 'speedup_percentiles' or None if
            insufficient data
        """
        if not covert_compute_data:
            return None

        years = covert_compute_data.get('years', [])
        operational = covert_compute_data.get('operational_black_project', {})

        # Build percentile dict from operational dark compute data
        covert_compute_percentiles = {}
        if 'p25' in operational:
            covert_compute_percentiles['p25'] = operational['p25']
        if 'median' in operational:
            covert_compute_percentiles['p50'] = operational['median']
        if 'p75' in operational:
            covert_compute_percentiles['p75'] = operational['p75']

        if not years or not covert_compute_percentiles:
            return None

        # Compute the compute cap
        compute_cap_data = self.compute_compute_cap(years, covert_compute_percentiles)

        if not compute_cap_data['compute']:
            return None

        # If we have a takeoff model, run trajectory prediction
        if takeoff_model is not None:
            if human_labor is None:
                # Default human labor - could be parameterized
                human_labor = [100.0] * len(years)

            self._trajectory = takeoff_model.predict_trajectory_deterministic(
                time=years,
                human_labor=human_labor,
                compute=compute_cap_data['compute']
            )
            if self._trajectory:
                self._trajectory['compute_cap'] = compute_cap_data
        else:
            # No takeoff model - just store compute cap
            self._trajectory = {'compute_cap': compute_cap_data}

        return self._trajectory

    @property
    def compute_cap(self) -> Optional[Dict[str, Any]]:
        """Get the computed compute cap."""
        return self._compute_cap

    @property
    def trajectory(self) -> Optional[Dict[str, Any]]:
        """Get the computed trajectory."""
        return self._trajectory

    def get_speedup_at_time(self, year: float) -> Optional[float]:
        """
        Get the AI R&D speedup at a specific time.

        Args:
            year: The year to query

        Returns:
            The AI R&D speedup at that time, or None if trajectory not computed
        """
        if self._trajectory is None or 'speedup' not in self._trajectory:
            return None

        years = self._trajectory.get('trajectory_times', [])
        speedups = self._trajectory.get('speedup', [])

        if not years or not speedups:
            return None

        # Interpolate to find speedup at the requested year
        return float(np.interp(year, years, speedups))