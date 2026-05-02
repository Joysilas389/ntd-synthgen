# Vector-borne NTDs

NTDs transmitted by arthropod vectors (mosquitoes, blackflies, sandflies, tsetse flies):

- **Lymphatic filariasis** (Wuchereria bancrofti)
- **Onchocerciasis** (river blindness)
- **Visceral and cutaneous leishmaniasis**
- **Human African trypanosomiasis** (sleeping sickness)

## Epidemiological context

Vector ecology is the dominant driver. Climate (temperature, humidity, rainfall) determines vector population dynamics. Land use and vegetation determine breeding and biting habitat. Bednet coverage and indoor residual spraying modulate human-vector contact.

## Schema design choices

`vegetation_index` is a normalized NDVI proxy on a `[0, 1]` scale. `vector_density` is per trap-night, which makes it comparable across surveillance designs. The target is monthly reported `disease_cases`.

## Suggested seed datasets

- WHO Global Programme to Eliminate Lymphatic Filariasis (GPELF)
- African Programme for Onchocerciasis Control (APOC) legacy data
- MODIS NDVI rasters (NASA EOSDIS)
- Malaria Atlas Project covariate stack (entomological proxies)
