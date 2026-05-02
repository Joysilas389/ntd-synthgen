# Skin NTDs

NTDs with primary cutaneous manifestations:

- **Buruli ulcer** (Mycobacterium ulcerans)
- **Leprosy** (Mycobacterium leprae)
- **Yaws** (Treponema pallidum subsp. pertenue)
- **Scabies** and other ectoparasitoses
- **Mycetoma**, chromoblastomycosis, and deep mycoses

## Epidemiological context

Skin NTDs cluster in households and along occupational exposure paths. Buruli ulcer in particular shows a strong association with proximity to slow-moving water bodies. Detection depends heavily on healthcare access and active case-finding programmes.

## Schema design choices

`proximity_to_water_km` captures the distance-decay typical of Buruli ulcer transmission. `healthcare_access_index` and `skin_contact_occupation` are deliberately included because they modulate both true incidence and reporting bias. Synthetic data inherits whatever reporting bias is present in the seed; the synthesizer cannot fix that.

## Suggested seed datasets

- WHO Global Leprosy Programme annual reports
- Buruli ulcer surveillance data (national programmes in Ghana, Cote d'Ivoire, Benin)
- Active case-finding survey microdata
