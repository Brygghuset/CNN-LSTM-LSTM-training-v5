
2025-09-17
  

#### Timeseries Input (300 tidssteg × 16 features):

| Ordning | **Vital Signs** (7 features)        | Enhet         | Förkortning i kod (standardiserat namn) | NaN Före första giltiga värde | NaN Efter första giltiga värde | NaN Efter sista giltiga värde                  | Vital DB datakälla                                                 | Unified Normalization range (Low, High) |     |
| ------- | ----------------------------------- | ------------- | --------------------------------------- | ----------------------------- | ------------------------------ | ---------------------------------------------- | ------------------------------------------------------------------ | --------------------------------------- | --- |
| 1       | Heart Rate                          | (BPM)         | `HR`                                    | Default (70)                  | Smart Forward Fill             | Default värde 10 min efter sista giltiga värde | `Solar8000/HR`                                                     | 20, 200                                 |     |
| 2       | Systolic Blood Pressure             | (mmHg)        | `BP_SYS`                                | Default (140)                 | Smart Forward Fill             | Default värde 10 min efter sista giltiga värde | `Solar8000/ART_SBP` `Solar8000/NIBP_SBP`                           | 60, 250                                 |     |
| 3       | Diastolic Blood Pressure            | (mmHg)        | `BP_DIA`                                | Default (80)                  | Smart Forward Fill             | Default värde 10 min efter sista giltiga värde | `Solar8000/ART_DBP` `Solar8000/NIBP_SBP`                           | 30, 150                                 |     |
| 4       | Mean Arterial Pressure              | (mmHg)        | `BP_MAP`                                | Beräknas från default         | Smart Forward Fill             | Default värde 10 min efter sista giltiga värde | `Solar8000/ART_MBP` (primär) `Solar8000/NIBP_MBP` `EV1000/ART_MBP` | 40, 180                                 |     |
| 5       | SpO2                                | (%)           | `SPO2`                                  | Default (96)                  | Smart Forward Fill             | Default värde 5 min efter sista giltiga värde  | `Solar8000/PLETH_SPO2`                                             | 70, 100                                 |     |
| 6       | EtCO2                               | (kPa)         | `ETCO2`                                 | Klinisk nolla (0.0)           | Smart Forward Fill             | Klinisk nolla (0.0)                            | `Solar8000/ETCO2`                                                  | 2.0, 8.0                                |     |
| 7       | BIS                                 | (Numerisk)    | `BIS`                                   | Klinisk nolla (0.0)           | Smart Forward Fill             | Klinisk nolla (0.0)                            | `BIS/BIS`                                                          | 0, 100                                  |     |
|         |                                     |               |                                         |                               |                                |                                                |                                                                    |                                         |     |
|         | **Drug Infusions (3 features)**:    |               |                                         |                               |                                |                                                |                                                                    |                                         |     |
| 8       | Propofol                            | (mg/kg/h)     | `Propofol_INF`                          | Klinisk nolla (0.0)           | Smart Forward Fill             | Klinisk nolla (0.0)                            | `Orchestra/PPF20_RATE`                                             | 0, 12                                   |     |
| 9       | Remifentanil                        | (mcg/kg/min)  | `Remifentanil_INF`                      | Klinisk nolla (0.0)           | Smart Forward Fill             | Klinisk nolla (0.0)                            | `Orchestra/RFTN20_RATE`                                            | 0, 0.8                                  |     |
| 10      | Noradrenalin                        | (mcg/kg/min)  | `Noradrenalin_INF`                      | Klinisk nolla (0.0)           | Klinisk nolla (0.0)            | Klinisk nolla (0.0)                            | `Orchestra/NEPI_RATE`                                              | 0, 0.5                                  |     |
|         |                                     |               |                                         |                               |                                |                                                |                                                                    |                                         |     |
|         | **Ventilator Settings (6 features** |               |                                         |                               |                                |                                                |                                                                    |                                         |     |
| 11      | Tidal Volume                        | (ml/kg IBW)   | `TV`                                    | Klinisk nolla (0.0)           | Smart Forward Fill             | Klinisk nolla (0.0)                            | `Solar8000/VENT_TV`                                                | 0, 12                                   |     |
| 12      | PEEP                                | (cmH2O)       | `PEEP`                                  | Klinisk nolla (0.0)           | Smart Forward Fill             | Klinisk nolla (0.0)                            | `Solar8000/VENT_MEAS_PEEP`                                         | 0, 30                                   |     |
| 13      | FiO2                                | (%)           | `FIO2`                                  | Klinisk nolla (0.0)           | Smart Forward Fill             | Klinisk nolla (0.0)                            | `Solar8000/FIO2`                                                   | 21, 100                                 |     |
| 14      | Respiratory Rate                    | (breaths/min) | `RR`                                    | Klinisk nolla (0.0)           | Smart Forward Fill             | Klinisk nolla (0.0)                            | `Solar8000/RR` `Solar8000/RR_CO2` `Primus/RR_CO2` (Primär)         | 6, 30                                   |     |
| 15      | Expiratory sevoflurane pressure     | (kPa)         | `etSEV`                                 | Klinisk nolla (0.0)           | Smart Forward Fill             | Klinisk nolla (0.0)                            | `Primus/EXP_SEVO`                                                  | 0, 6                                    |     |
| 16      | Inspiratory sevoflurane pressure    | (kPa)         | `inSev`                                 | Klinisk nolla (0.0)           | Smart Forward Fill             | Klinisk nolla (0.0)                            | `Primus/INSP_SEVO`                                                 | 0, 8                                    |     |


#### Static Patient Features (6 features):

| Ordning | Static Patient Features | Förkortning i kod (standardiserat namn) | Enhet   | Default-värde | Range                   | Formel för normalization |
| ------- | ----------------------- | --------------------------------------- | ------- | ------------- | ----------------------- | ------------------------ |
| 17      | Age                     | `age`                                   | (years) | 50            | 0-120 år                | age/120 × 2 - 1          |
| 18      | Sex                     | `sex`                                   | (F/M)   | -1            | (-1 = Female, 1 = Male) | 1.0 eller -1.0           |
| 19      | Height                  | `height`                                | (cm)    | 170           | 100-230 cm              | (height-100)/130 × 2 - 1 |
| 20      | Weight                  | `weight`                                | (kg)    | 70            | 20-200 kg               | (weight-20)/180 × 2 - 1  |
| 21      | BMI                     | `bmi`                                   | (kg/m²) | 24,2          | 10-50 BMI               | (bmi-10)/40 × 2 - 1      |
| 22      | ASA Score               | `asa`                                   | (1-6)   | 2             | ASA 1-6                 | (asa-1)/5 × 2 - 1        |

#### Output

| Ordning | Drug Output (3 predictions)           | Enhet         | Förkortning i kod (standardiserat namn) | Range   | inverse normalization formula |
| ------- | ------------------------------------- | ------------- | --------------------------------------- | ------- | ----------------------------- |
| 1       | Propofol                              | (mg/kg/h)     | `Propofol_Predict`                      | 0, 12   | (norm + 1) × 6                |
| 2       | Remifentanil                          | (mcg/kg/min)  | `Remifentanil_Predict`                  | 0, 0.8  | (norm + 1) × 0.4              |
| 3       | Noradrenalin                          | (mcg/kg/min)  | `Noradrenalin_Predict`                  | 0, 0.5  | (norm + 1) × 0.25             |
|         | **Ventilator Output (5 predictions)** |               |                                         |         |                               |
| 4       | Tidal Volume                          | (ml/kg IBW)   | `TV_Predict`                            | 0, 12   | (norm + 1) × 6                |
| 5       | PEEP                                  | (cmH2O)       | `PEEP_Predict`                          | 0, 30   | (norm + 1) × 15               |
| 6       | FiO2                                  | (%)           | `FIO2_Predict`                          | 21, 100 | 21 + (norm + 1) × 39.5        |
| 7       | Respiratory Rate                      | (breaths/min) | `RR_Predict`                            | 6, 30   | 6 + (norm + 1) × 12           |
| 8       | Expiratory sevoflurane pressure       | (kPa)         | `etSEV_Predict`                         | 0, 6    | (norm + 1) × 3                |


  
  

------




#### Fill
Mean-värden: Beräknas från träningsdata (series.mean())
Forward fill: Använder tidigare värden från samma serie
Backward fill: Använder senare värden från samma serie
#### Från DataPreprocessor - Smart Forward Fill:
def _smart_forward_fill(self, series, max_consecutive_nans=None):
    # Steg 1: Identifiera isolerade NaN
    isolated_nans = []
    for i in range(1, len(series) - 1):
        if pd.isna(series.iloc[i]) and not pd.isna(series.iloc[i-1]) and not pd.isna(series.iloc[i+1]):
            isolated_nans.append(i)
    
    # Steg 2: Mean-imputation för isolerade NaN
    if isolated_nans:
        mean_value = series.mean()  # ← FRÅN TRÄNINGSDATA
        for idx in isolated_nans:
            imputed.iloc[idx] = mean_value
    
    # Steg 3: Forward fill för kvarvarande NaN (inklusive initiala)
    imputed = imputed.ffill(limit=max_consecutive_nans)
    
    # Steg 4: Backward fill som fallback för initiala NaN
    if imputed.isna().any():
        imputed = imputed.bfill()



###  Formeln för Unified Normalization
normalized_value = (value - min_clinical) / (max_clinical - min_clinical) × (target_max - target_min) + target_min

- value = Ursprungligt värde
- min_clinical = Klinisk minimum-gräns för parametern
- max_clinical = Klinisk maximum-gräns för parametern
- target_min = Mål-intervall minimum (t.ex. -1)
- target_max = Mål-intervall maximum (t.ex. 1)

#### Reverse normalization formula: 
- För ranges som börjar på 0: `(norm + 1) × (max/2)`
- För ranges som inte börjar på 0: `min + (norm + 1) × ((max-min)/2)`

Tips för framtida Unified Normalization range: Analysera träningsdatan och justera range så att alla extremvärden täcks in, lägg därefter till 10-20% marginal upp/ner för nya extremvärden. 