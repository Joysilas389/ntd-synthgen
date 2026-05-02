# Water-borne NTDs

NTDs transmitted through contaminated water and poor sanitation:

- **Schistosomiasis** (bilharzia)
- **Soil-transmitted helminthiases** (ascariasis, hookworm, trichuriasis)
- **Dracunculiasis** (Guinea worm disease)

## Epidemiological context

In Sub-Saharan Africa, transmission risk for these diseases is mostly driven by access to safe water sources, sanitation coverage, rainfall (which affects surface water and the habitat of intermediate hosts like snails), and population density and contact patterns.

## Schema design choices

`sanitation_index` is a composite of household-level WASH indicators on a `[0, 1]` scale. `open_defecation_rate` is the proportion of the population practicing open defecation. The target variable is monthly `disease_cases` per administrative unit.

## Suggested seed datasets

- WHO Preventive Chemotherapy and Transmission Control databank
- UNICEF/WHO Joint Monitoring Programme for WASH
- National DHS (Demographic and Health Surveys)
