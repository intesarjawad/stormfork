import xarray as xr

def extract_inputs_targets_forcings(
    dataset: xr.Dataset,
    *,
    input_variables: tuple[str, ...],
    target_variables: tuple[str, ...],
    forcing_variables: tuple[str, ...],
    pressure_levels: tuple[int, ...],
    input_duration: str,
    target_lead_times: slice,
) -> tuple[xr.Dataset, xr.Dataset, xr.Dataset]: ...
