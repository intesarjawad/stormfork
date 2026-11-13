from weathernext.utils.fiddle_config_io import get_fiddle_config_by_name

ATMOSPHERE = (
    "temperature",
    "geopotential",
    "u_component_of_wind",
    "v_component_of_wind",
    "vertical_velocity",
    "specific_humidity",
)
SURFACE = (
    "2m_temperature",
    "mean_sea_level_pressure",
    "10m_v_component_of_wind",
    "10m_u_component_of_wind",
    "sea_surface_temperature",
)
STATICS = ("geopotential_at_surface", "land_sea_mask")
FORCINGS = ("year_progress_sin", "year_progress_cos", "day_progress_sin", "day_progress_cos")
CYCLONE_TARGETS = (
    "cyclone_exists_gaussian_unit_mode",
    "cyclone_all_wind_disc",
    "cyclone_usa_wind_disc",
    "cyclone_usa_r34_ne_radius_disc",
    "cyclone_usa_r34_se_radius_disc",
    "cyclone_usa_r34_sw_radius_disc",
    "cyclone_usa_r34_nw_radius_disc",
    "cyclone_usa_r50_ne_radius_disc",
    "cyclone_usa_r50_se_radius_disc",
    "cyclone_usa_r50_sw_radius_disc",
    "cyclone_usa_r50_nw_radius_disc",
    "cyclone_usa_r64_ne_radius_disc",
    "cyclone_usa_r64_se_radius_disc",
    "cyclone_usa_r64_sw_radius_disc",
    "cyclone_usa_r64_nw_radius_disc",
    "cyclone_usa_rmw_disc",
    "cyclone_usa_pres_disc",
)


def test_mini_input_contract():
    task = get_fiddle_config_by_name("weathernext2/configs/WeatherNextCyclones_Mini").task
    assert task.input_variables == ATMOSPHERE + SURFACE + STATICS + FORCINGS
    assert (
        task.target_variables
        == ATMOSPHERE + SURFACE + ("total_precipitation_6hr",) + CYCLONE_TARGETS
    )
    assert task.forcing_variables == FORCINGS
    assert task.pressure_levels == (50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000)
    assert task.input_duration == "12h"
    assert not any(name.startswith("cyclone_") for name in task.input_variables)
    assert "total_precipitation_6hr" not in task.input_variables
