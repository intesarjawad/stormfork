# Licences and attribution

The repository's original code is MIT licensed. Third-party data and model terms apply separately.

| Material | Terms and attribution |
|---|---|
| WN2 recorded fixture and collected forecasts | CC BY 4.0. Weather data by [Open-Meteo.com](https://open-meteo.com/). The free API permits non-commercial use under its request limits. |
| ASOS station metadata | Retrieved from the [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/). Source URLs and retrieval dates are in `tests/fixtures/towns_source.json`. |
| WeatherNext source dependency | Apache-2.0, as declared by the pinned source package. |
| WeatherNext Cyclones Mini weights | CC BY 4.0. "WeatherNext Cyclones, Google DeepMind, CC BY 4.0." Downloaded by `stormfork fetch` from Google DeepMind's bucket; not redistributed here. |
| HRES sample | ECMWF HRES data under CC BY 4.0, as modified by Google. Required statement: Copyright "© 2023 European Centre for Medium-Range Weather Forecasts (ECMWF)". Source: [www.ecmwf.int](https://www.ecmwf.int/). Disclaimer: ECMWF does not accept any liability whatsoever for any error or omission in the data, their availability, or for any loss or damage arising from their use. Terms from the Acknowledgements of the [weathernext v0.3.0 README](https://github.com/google-deepmind/weathernext/blob/v0.3.0/README.md), read 2026-11-09. Every manifest credits it, because every initial-condition source takes terrain and land mask from the sample. Downloaded by `stormfork fetch`; not redistributed here. |
| Site fonts | IM Fell English, Igino Marini; the figures of EB Garamond, Octavio Pardo and the EB Garamond Project Authors; and Departure Mono, Helena Zhang; all under the SIL Open Font License 1.1. The licence texts are in `site/src/style/fonts/`. |
| IBTrACS v04r01 | NOAA NCEI, US government work, public domain. Cite Knapp et al. 2010. Downloaded by `stormfork fetch`; not redistributed here. |

The recorded WN2 request and metadata endpoint are in `tests/fixtures/wn2_source.json`.
