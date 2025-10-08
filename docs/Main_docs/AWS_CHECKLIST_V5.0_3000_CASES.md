# AWS Preprocessing Checklist v5.0 - 3000 Cases Implementation

**VIKTIGT**: Känslig AWS-konfiguration finns i `aws_config.env` filen. 
Se denna fil för aktuella bucket-namn, S3-paths och AWS-konfigurationer.

**Mål**: Framgångsrik preprocessing av 3000 cases med full funktionalitet  
**Specifikation**: Master POC CNN-LSTM-LSTM.md  
**Baserad på**: Analys av fungerande (2051 cases) och misslyckade (3000 cases) körningar  
**Skapad**: 2025-10-07

---

## Executive Summary

### Kritiska Fynd från Analys
1. ✅ **2051 case körning (Juli 2025)** - FUNGERADE med `pytorch_preprocessing_entry_point.py`
2. ❌ **200/20 case körningar (Okt 2025)** - SAG lyckade ut men skapade INGA TFRecord-filer
3. ❌ **3000 case körning (Okt 2025)** - Totalt misslyckande med dubbel processing + ingen output

### Root Causes
- **Software Regression**: Fungerande implementation ersatt med ofullständig version
- **Ingen Output Validation**: Success i loggar trots att inga TFRecord-filer skapades
- **Ingen Multi-Instance Distribution**: Alla instanser processar samma cases parallellt
- **Spot Instance Outestade**: Första gången använt, checkpoint-system fungerade inte

---

## TABELL 1: Master POC Specifikation (från Master POC CNN-LSTM-LSTM.md)

| Kategori | Parameter | Specifikation | Obligatorisk | Kommentar |
|----------|-----------|---------------|--------------|-----------|
| **INPUT FEATURES** | Timeseries Features | **16 features** | ✅ JA | 7 vital signs + 3 drugs + 6 ventilator |
| | Static Features | **6 features** | ✅ JA | age, sex, height, weight, bmi, asa |
| | Total Input | **22 features** (16+6) | ✅ JA | Enligt Master POC spec |
| **OUTPUT FEATURES** | Drug Predictions | **3 predictions** | ✅ JA | Propofol, Remifentanil, Noradrenalin |
| | Ventilator Predictions | **5 predictions** | ✅ JA | TV, PEEP, FIO2, RR, etSEV |
| | Total Output | **8 predictions** | ✅ JA | 3 drugs + 5 ventilator |
| **NORMALIZATION** | Range | **[-1, 1]** | ✅ JA | Unified normalization |
| | Method | **Unified Normalization Formula** | ✅ JA | `(value - min) / (max - min) × 2 - 1` |
| **WINDOW** | Window Size | **300 sekunder** (5 min) | ✅ JA | Sliding window |
| | Step Size | **30 sekunder** | ✅ JA | 10% overlap |
| **IMPUTATION** | Method | **Smart Forward Fill** | ✅ JA | Kliniska nollor + forward fill |
| | Clinical Zeros | Definierade per parameter | ✅ JA | Se Master POC spec |
| **UNIT CONVERSION** | Drugs | **mg/kg/h, μg/kg/min** | ✅ JA | Från mL/h med koncentration |
| | Tidal Volume | **ml/kg IBW** | ✅ JA | Med Devine formula |
| | Pressure | **kPa, cmH2O** | ✅ JA | Unit conversions |

---

## TABELL 2: Funktionalitet & Bästa Implementation

| Funktionalitet | Prioritet | Bästa Implementation | Källa Fil | Status i v5.0 | Kritiska Noter |
|----------------|-----------|---------------------|-----------|---------------|----------------|
| **CORE PROCESSING** |
| Case Range Parsing | 🔴 KRITISK | Batch + Comma-separated | `pytorch_preprocessing_entry_point.py` rad 146-193 | ✅ IMPLEMENT | Stödjer "1-3000", "1,2,3" men INTE mixed |
| Master POC Feature Mapping | 🔴 KRITISK | 16 timeseries features | `master_poc_preprocessing_entry_point.py` + orchestrator | ✅ IMPLEMENT | MÅSTE följa Master POC spec |
| Unit Conversion | 🔴 KRITISK | Drug concentrations + IBW | `master_poc_preprocessing_orchestrator.py` | ✅ IMPLEMENT | 20mg/ml Propofol, 20μg/ml Remifentanil |
| Smart Forward Fill | 🔴 KRITISK | Clinical zeros + forward fill | Master POC spec tabell | ✅ IMPLEMENT | Olika regler per parameter |
| Unified Normalization | 🔴 KRITISK | [-1, 1] range | Master POC spec formel | ✅ IMPLEMENT | Med clinical min/max ranges |
| Window Creation | 🔴 KRITISK | 300s windows, 30s step | Både implementations | ✅ IMPLEMENT | Sliding windows |
| Static Features | 🔴 KRITISK | 6 features normalized | `pytorch_preprocessing_entry_point.py` rad 1365-1402 | ✅ IMPLEMENT | age, sex, height, weight, bmi, asa |
| **OUTPUT & STORAGE** |
| TFRecord Creation | 🔴 KRITISK | Memory-efficient streaming | `pytorch_preprocessing_entry_point.py` rad 1761-1827 | ✅ IMPLEMENT | SAKNAS i master_poc entry point |
| Train/Val/Test Split | 🔴 KRITISK | 70/15/15 split | `pytorch_preprocessing_entry_point.py` rad 1765-1789 | ✅ IMPLEMENT | 3 separata TFRecord-filer |
| Incremental TFRecord Save | 🟡 HOOG | Per case eller batch | `master_poc_preprocessing_orchestrator.py` rad 398-504 | 🆕 NY | Spara löpande, inte vid completion |
| S3 Upload | 🔴 KRITISK | Automatisk efter completion | `pytorch_preprocessing_entry_point.py` rad 1435-1447 | ✅ IMPLEMENT | MÅSTE ske innan timeout |
| Metadata Save | 🟡 HOOG | JSON med full statistik | `pytorch_preprocessing_entry_point.py` rad 1791-1804 | ✅ IMPLEMENT | Lokal + S3 |
| **ROBUSTNESS** |
| Checkpoint Creation | 🔴 KRITISK | PreprocessingCheckpointManager | `checkpoint_manager.py` rad 48-361 | ✅ IMPLEMENT | SAKNAS i master_poc entry point |
| Checkpoint Resume | 🔴 KRITISK | Initialize + resume logic | `pytorch_preprocessing_entry_point.py` rad 1614-1630 | ✅ IMPLEMENT | Vid spot instance restart |
| Enable Checkpoints Flag | 🔴 KRITISK | `--enable-checkpoints` | Båda entry points | ✅ IMPLEMENT | MÅSTE sättas till TRUE |
| Checkpoint Interval | 🟡 HOOG | Var 10-50:e case | Argument | ✅ IMPLEMENT | Balans mellan overhead & safety |
| Error Handling | 🟡 HOOG | Try/except per case | Båda implementations | ✅ IMPLEMENT | Logga fel, fortsätt processing |
| Memory Management | 🟡 HOOG | Memory-efficient batch processor | `pytorch_preprocessing_entry_point.py` rad 1093-1403 | ✅ IMPLEMENT | För stora datasets |
| **MULTI-INSTANCE** |
| Case Distribution | 🔴 KRITISK | Dela cases mellan instanser | ❌ **SAKNAS HELT** | 🆕 NY | **KRITISKT SAKNAS** |
| Host Detection | 🔴 KRITISK | SM_CURRENT_HOST env var | SageMaker environment | 🆕 NY | Identifiera vilken instans |
| Host Count Detection | 🔴 KRITISK | SM_HOSTS env var | SageMaker environment | 🆕 NY | Total antal instanser |
| Case Partitioning Logic | 🔴 KRITISK | Modulo eller range-based | 🆕 **BEHÖVER SKAPAS** | 🆕 NY | Ex: instance_id % total_instances |
| Distributed Checkpoint | 🟡 HOOG | Per-instance checkpoints | 🆕 **BEHÖVER SKAPAS** | 🆕 NY | S3 path med instance ID |
| **SPOT INSTANCE** |
| Spot Instance Support | 🟡 HOOG | `use_spot_instances=True` | Estimator config | ✅ IMPLEMENT | 65-70% kostnadsbesparing |
| Max Wait Time | 🟡 HOOG | `max_wait = max_run * 2` | Estimator config | ✅ IMPLEMENT | Flexibilitet för spot restarts |
| Spot Interrupt Handling | 🔴 KRITISK | Checkpoint + resume | Checkpoint manager | ✅ IMPLEMENT | Måste fungera med checkpoints |
| **VALIDATION** |
| Output Verification | 🔴 KRITISK | Verifiera TFRecord-filer | 🆕 **BEHÖVER SKAPAS** | 🆕 NY | Kontrollera att filer faktiskt skapas |
| S3 Verification | 🔴 KRITISK | Lista filer i S3 | 🆕 **BEHÖVER SKAPAS** | 🆕 NY | Verifiera upload lyckades |
| Window Count Validation | 🟡 HOOG | Förväntat vs faktiskt | Metadata | ✅ IMPLEMENT | Sanity check |
| Feature Count Validation | 🟡 HOOG | 16+6 input, 8 output | Pre-flight check | ✅ IMPLEMENT | Verifiera spec compliance |

---

## TABELL 3: Kritiska Saknade Funktioner (MÅSTE Implementeras)

| Funktion | Problem | Impact | Lösning | Prioritet | Uppskattad Komplexitet |
|----------|---------|--------|---------|-----------|------------------------|
| **Multi-Instance Case Distribution** | Alla instanser processar SAMMA cases | 6x redundant arbete, konflikt vid S3 upload | Implementera case partitioning baserat på `SM_CURRENT_HOST` och `SM_HOSTS` | 🔴 KRITISK | 🟢 LÅG (50 rader kod) |
| **Incremental TFRecord Save** | Data sparas endast vid completion | ALL data förloras vid timeout | Spara TFRecord löpande per batch eller case | 🔴 KRITISK | 🟡 MEDEL (100 rader kod) |
| **Enable Checkpoints Default** | Checkpoints aldrig aktiverade | Ingen resume möjlighet | Sätt `default=True` eller inkludera i starter script | 🔴 KRITISK | 🟢 LÅG (1 rad kod) |
| **TFRecord Integration i Master POC** | Entry point sparar INTE TFRecord | Ingen output från processing | Integrera `create_memory_efficient_tfrecord()` | 🔴 KRITISK | 🟡 MEDEL (150 rader kod) |
| **Output Verification** | Ingen validering att filer skapades | False positives i success loggar | Lägg till S3 list + count check | 🔴 KRITISK | 🟢 LÅG (30 rader kod) |
| **Distributed Checkpoint Paths** | Alla instanser skriver till samma checkpoint | Konflikt och corruption | Unik checkpoint path per instance | 🟡 HOOG | 🟢 LÅG (20 rader kod) |
| **Spot Interrupt Testing** | Aldrig testat med spot instances | Okända failure modes | Testköring med spot + manuell interrupt | 🟡 HOOG | 🟡 MEDEL (test körning) |
| **Train/Val/Test Split Logic** | Saknas i master_poc entry point | Ingen split av data | Kopiera från pytorch_preprocessing | 🔴 KRITISK | 🟢 LÅG (30 rader kod) |
| **Robust Case Format Parsing** | Stödjer inte blandat format "1-10,17,0022" | Begränsad flexibilitet i case selection | Utöka parse_case_range() med mixed format support | 🔴 KRITISK | 🟢 LÅG (40 rader kod) |
| **Graceful Shutdown Handling** | Ingen SIGTERM hantering vid timeout | Data förloras vid MaxRuntimeExceeded | Implementera signal handler för graceful shutdown | 🔴 KRITISK | 🟡 MEDEL (80 rader kod) |
| **S3 Retry Logic** | Inga retries vid S3 failures | Spot instance restarts kan orsaka S3 fel | Exponential backoff för alla S3 operations | 🔴 KRITISK | 🟡 MEDEL (60 rader kod) |
| **Idempotent S3 Writes** | Återkörningar kan skapa dubletter | Konflikt vid multiple runs | Overwrite policy eller unique job_id suffix | 🟡 HOOG | 🟢 LÅG (20 rader kod) |

---

## TABELL 4: AWS SageMaker Configuration för 3000 Cases

| Parameter | Värde | Källa | Motivering | Validering |
|-----------|-------|-------|------------|------------|
| **ESTIMATOR CONFIG** |
| Entry Point | `master_poc_preprocessing_v5.py` | 🆕 NY FIL | Kombinerar Master POC spec + fungerande funktioner | Måste skapas |
| Source Dir | `src/` | Standard | All källkod i src/ | Befintlig struktur |
| Framework Version | `1.12.1` | AWS Guide v4.1 | PyTorch framework | Testad |
| Python Version | `py38` | AWS Guide v4.1 | Python 3.8 | Testad |
| Instance Type | `ml.m5.2xlarge` | Guide v4.1 + Master POC | 8 vCPU, 32GB RAM | Adequate för processing |
| Instance Count | `6` | Master POC Instruction v1.1 | Parallellisering | **KRÄVER case distribution** |
| Use Spot Instances | `True` | Master POC Instruction v1.1 | 65-70% kostnadsbesparing | Testat (men failed) |
| Max Run | `93600s` (26h) | Uppskattning | 3000 cases × ~30s/case / 6 inst | Konservativ uppskattning |
| Max Wait | `187200s` (52h) | `max_run × 2` | Spot flexibilitet | Master POC Instruction |
| **HYPERPARAMETERS** |
| Cases | `1-3000` | Requirement | Full dataset | Standard format |
| Batch Size | `50` | Balans memory/checkpoint | Checkpoint var 50:e case | Testad |
| Checkpoint Interval | `50` | Balans overhead/safety | Var 50:e case | Master POC Instruction |
| **Enable Checkpoints** | `True` | 🆕 **ÄNDRING** | **MÅSTE vara True** | **Kritisk ändring** |
| Window Size | `300` | Master POC spec | 5 minuters fönster | Specifikation |
| Step Size | `30` | Master POC spec | 30 sekunders steg | Specifikation |
| Timeseries Features | `16` | Master POC spec | **INTE 14** | Specifikation |
| Static Features | `6` | Master POC spec | age, sex, height, weight, bmi, asa | Specifikation |
| Output Features | `8` | Master POC spec | **INTE 7** | Specifikation |
| Normalization Range | `-1.0 1.0` | Master POC spec | Mellanslag, INTE komma | Specifikation |
| Imputation Method | `master_poc_smart_forward_fill` | Master POC spec | Clinical zeros + forward fill | Specifikation |
| Pipeline Type | `master_poc` | Identifier | Master POC pipeline | Standard |
| S3 Bucket | [Se `S3_PRIMARY_BUCKET` i `aws_config.env`] | Master POC Instruction | Data storage | Befintlig bucket |
| **S3 PATHS** |
| Input Data | [Se `S3_INPUT_PATH` i `aws_config.env`] | Master POC Instruction | VitalDB data | Befintlig |
| Output Path | [Se `S3_OUTPUT_PATH` i `aws_config.env`] | Master POC Instruction | TFRecord output | Befintlig |
| Checkpoint Path | [Se `S3_CHECKPOINT_PATH` i `aws_config.env`] | 🆕 NY | Checkpoint storage | Behöver skapas |

---

## TABELL 5: Implementation Plan

| Steg | Task | Beskrivning | Beroenden | Uppskattad Tid | Prioritet |
|------|------|-------------|-----------|----------------|-----------|
| **FAS 1: KRITISKA FIXES** |
| 1.1 | Skapa ny entry point | `master_poc_preprocessing_v5.py` kombinerar båda implementations | Ingen | 4 timmar | 🔴 KRITISK |
| 1.2 | Implementera case distribution | Dela cases mellan instanser med SM_CURRENT_HOST | 1.1 | 2 timmar | 🔴 KRITISK |
| 1.3 | Integrera TFRecord creation | Kopiera från pytorch_preprocessing | 1.1 | 2 timmar | 🔴 KRITISK |
| 1.4 | Integrera checkpoint manager | Kopiera från pytorch_preprocessing | 1.1 | 2 timmar | 🔴 KRITISK |
| 1.5 | Implementera incremental save | TFRecord save per batch | 1.3, 1.4 | 3 timmar | 🔴 KRITISK |
| 1.6 | Enable checkpoints default | Sätt default=True | 1.4 | 5 minuter | 🔴 KRITISK |
| 1.7 | Lägg till output verification | S3 list + count check | 1.3 | 1 timme | 🔴 KRITISK |
| **FAS 2: TESTER** |
| 2.1 | Test case distribution | Kör 10 cases med 2 instanser | 1.2 | 30 minuter | 🔴 KRITISK |
| 2.2 | Test checkpoint resume | Avbryt + resume test | 1.4 | 30 minuter | 🔴 KRITISK |
| 2.3 | Test incremental save | Verifiera TFRecord skapas löpande | 1.5 | 30 minuter | 🔴 KRITISK |
| 2.4 | Test spot instance | 50 cases med spot + manuell interrupt | 1.4, 1.6 | 1 timme | 🟡 HOOG |
| 2.5 | Test full pipeline | 100 cases med alla features | Alla ovan | 1 timme | 🟡 HOOG |
| **FAS 3: SMÅSKALIG PILOT** |
| 3.1 | Pilot 200 cases | Verifiera med 200 cases, 2 instanser | Fas 1-2 | 2 timmar | 🟡 HOOG |
| 3.2 | Verifiera output | Kontrollera TFRecord-filer i S3 | 3.1 | 30 minuter | 🟡 HOOG |
| 3.3 | Analysera metadata | Verifiera windows, features, split | 3.1 | 30 minuter | 🟡 HOOG |
| **FAS 4: FULLSKALIG KÖRNING** |
| 4.1 | 3000 case preprocessing | Full körning med 6 instanser, spot | Fas 1-3 | 26 timmar | 🟢 MÅLET |
| 4.2 | Övervaka progress | CloudWatch monitoring | 4.1 | Kontinuerlig | 🟢 MÅLET |
| 4.3 | Verifiera completion | Kontrollera alla TFRecord-filer | 4.1 | 1 timme | 🟢 MÅLET |
| 4.4 | Uppdatera dokumentation | Dokumentera v5.0 implementation | 4.3 | 2 timmar | 🟢 MÅLET |

**Total Uppskattad Tid**: 
- Fas 1-2: ~15 timmar development + testing
- Fas 3: ~3 timmar pilot
- Fas 4: ~26 timmar körning + 3 timmar verifiering

---

## TABELL 6: Code Snippets - Multi-Instance Distribution

### Implementation av Case Distribution (NYA FUNKTIONER)

```python
# I master_poc_preprocessing_v5.py

import os
import json

def get_sagemaker_host_info():
    """Hämta SageMaker host information från environment variables."""
    training_env = json.loads(os.environ.get('SM_TRAINING_ENV', '{}'))
    
    current_host = training_env.get('current_host', 'algo-1')
    all_hosts = training_env.get('hosts', ['algo-1'])
    host_index = all_hosts.index(current_host)
    total_hosts = len(all_hosts)
    
    logger.info(f"🖥️ SageMaker Host Info:")
    logger.info(f"   Current Host: {current_host} (index {host_index})")
    logger.info(f"   Total Hosts: {total_hosts}")
    logger.info(f"   All Hosts: {all_hosts}")
    
    return {
        'current_host': current_host,
        'host_index': host_index,
        'total_hosts': total_hosts,
        'all_hosts': all_hosts
    }

def distribute_cases_across_instances(case_ids: List[str], host_info: dict) -> List[str]:
    """
    Dela upp cases mellan instanser för att undvika dubbel processing.
    
    Använder modulo för jämn fördelning:
    - Instance 0 får case 0, 6, 12, 18, ...
    - Instance 1 får case 1, 7, 13, 19, ...
    - Instance 2 får case 2, 8, 14, 20, ...
    etc.
    """
    host_index = host_info['host_index']
    total_hosts = host_info['total_hosts']
    
    # Filtrera cases för denna instans
    instance_cases = [
        case_id for i, case_id in enumerate(case_ids)
        if i % total_hosts == host_index
    ]
    
    logger.info(f"📊 Case Distribution:")
    logger.info(f"   Total Cases: {len(case_ids)}")
    logger.info(f"   This Instance Cases: {len(instance_cases)}")
    logger.info(f"   First 5: {instance_cases[:5]}")
    logger.info(f"   Last 5: {instance_cases[-5:]}")
    
    return instance_cases

def create_distributed_checkpoint_path(base_path: str, host_info: dict) -> str:
    """Skapa unik checkpoint path för varje instans."""
    current_host = host_info['current_host']
    checkpoint_path = f"{base_path}/{current_host}"
    
    logger.info(f"💾 Checkpoint Path: {checkpoint_path}")
    return checkpoint_path

# Användning i main():
def main():
    args = parse_args()
    
    # Hämta host info
    host_info = get_sagemaker_host_info()
    
    # Parse alla cases
    all_case_ids = parse_case_range(args.cases)
    
    # Distribuera cases till denna instans
    instance_cases = distribute_cases_across_instances(all_case_ids, host_info)
    
    # Skapa instans-specifik checkpoint path
    checkpoint_path = create_distributed_checkpoint_path(
        args.checkpoint_base_path,
        host_info
    )
    
    # Processa endast denna instans cases
    process_cases(instance_cases, checkpoint_path, args)
```

---

## TABELL 7: Verifierings-Checklista (Efter Körning)

| Verifiering | Kommando/Metod | Förväntat Resultat | Kritiskt? |
|-------------|----------------|-------------------|-----------|
| **S3 OUTPUT** |
| TFRecord-filer skapade | `aws s3 ls s3://master-poc-v1.0/processed-data/master-poc-pipeline/ --recursive` | 3 filer: train.tfrecord, validation.tfrecord, test.tfrecord | ✅ JA |
| TFRecord-storlek | `aws s3 ls s3://... --human-readable` | >100MB per fil (beroende på cases) | ✅ JA |
| Metadata fil | `aws s3 ls ...preprocessing_metadata.json` | 1 fil med statistik | ✅ JA |
| **METADATA INNEHÅLL** |
| Total samples | Läs JSON: `total_samples` | >100,000 windows (beroende på cases) | ✅ JA |
| Train/val/test split | Läs JSON: `train_samples`, `validation_samples`, `test_samples` | 70/15/15 split | ✅ JA |
| Window shape | Läs JSON: `window_shape` | [300, 16] | ✅ JA |
| Target shape | Läs JSON: `target_shape` | [8] | ✅ JA |
| Static shape | Läs JSON: `static_features_shape` | [6] | ✅ JA |
| Success rate | Läs JSON: `success_rate` | >95% | ⚠️ ÖNSKVÄRT |
| **CHECKPOINT** |
| Checkpoint-filer | `aws s3 ls s3://master-poc-v1.0/checkpoints/master-poc-v5-3000cases/ --recursive` | 6 filer (en per instans) | ⚠️ VID SPOT |
| Checkpoint state | Läs checkpoint JSON | processed_cases, failed_cases lists | ⚠️ VID SPOT |
| **LOGS** |
| CloudWatch loggar | AWS Console eller CLI | Success meddelanden, inga errors | ✅ JA |
| Processing time | CloudWatch metrics | ~26 timmar total | ⚠️ UPPSKATTNING |
| Cost | AWS Cost Explorer | ~$12-15 med spot instances | ⚠️ INFO |
| **FUNKTIONALITET** |
| Case distribution | Sök i loggar: "This Instance Cases" | Olika antal per instans | ✅ JA |
| No double processing | Verifiera case overlap mellan instanser | Ingen overlap | ✅ JA |
| Checkpoint saves | Sök i loggar: "Checkpoint saved" | Var 50:e case | ⚠️ VID SPOT |
| TFRecord saves | Sök i loggar: "TFRecord files created" | Vid completion | ✅ JA |

---

## TABELL 8: Risk Assessment & Mitigation

| Risk | Sannolikhet | Impact | Mitigation | Contingency Plan |
|------|-------------|--------|------------|------------------|
| **Spot Instance Interrupt** | 🟡 MEDEL | 🟡 MEDEL | Checkpoint var 50:e case + max_wait 52h | Resume från checkpoint automatiskt |
| **Timeout (26h inte tillräckligt)** | 🟡 MEDEL | 🔴 HOOG | Incremental save + konservativ tid-uppskattning | Förläng max_run till 36h |
| **Instance case overlap** | 🔴 HOOG (om ej fixat) | 🔴 KRITISK | Case distribution implementation | Kör endast 1 instans som fallback |
| **TFRecord corruption** | 🟢 LÅG | 🔴 HOOG | Validera efter varje save | Checkpoint resume + reprocess |
| **S3 upload failure** | 🟢 LÅG | 🔴 HOOG | Retry logic + verification | Manuell upload från checkpoint |
| **Memory overflow** | 🟡 MEDEL | 🟡 MEDEL | Memory-efficient batch processor + batch size 50 | Minska batch size till 25 |
| **Feature mapping fel** | 🟢 LÅG | 🔴 KRITISK | Pre-flight validation + unit tests | Debugga + rerun med fix |
| **Cost overrun** | 🟢 LÅG | 🟡 MEDEL | Spot instances (-70%) + monitoring | Avbryt jobb om kostnad >$20 |

---

## SAMMANFATTNING: Kritiska Ändringar från Tidigare

### ✅ Fungerande (Behåll från 2051 case körning)
- `pytorch_preprocessing_entry_point.py` structure
- TFRecord creation med train/val/test split
- Checkpoint manager integration
- Memory-efficient processing
- S3 upload automation

### ✅ Fungerande (Behåll från Master POC)
- 16+6 features (INTE 14+6)
- 8 outputs (INTE 7)
- Master POC unit conversion
- Smart forward fill med clinical zeros
- Unified normalization [-1, 1]

### 🆕 NYA Implementationer (KRITISKA)
1. **Multi-Instance Case Distribution** - Dela cases mellan 6 instanser
2. **Incremental TFRecord Save** - Spara löpande, inte vid completion
3. **Enable Checkpoints Default** - Sätt till True
4. **Output Verification** - Verifiera att filer faktiskt skapas
5. **Distributed Checkpoint Paths** - Unika paths per instans

### ❌ TA BORT (Från tidigare körningar)
- ❌ Single-instance assumption
- ❌ Bulk save endast vid completion
- ❌ Checkpoint disabled by default
- ❌ Ingen output validation

---

## NÄSTA STEG

1. **Skapa ny entry point**: `master_poc_preprocessing_v5.py`
2. **Implementera case distribution**: Kod snippets ovan
3. **Test med 10 cases, 2 instanser**: Verifiera distribution
4. **Test checkpoint resume**: Manuell interrupt + resume
5. **Pilot 200 cases**: Full validation
6. **3000 cases production run**: Med monitoring

**Estimerat: 2-3 dagars utveckling + tester innan production körning**

---

**Version**: 5.0  
**Skapad**: 2025-10-07  
**Status**: Redo för implementation  
**Next Review**: Efter pilot 200 cases

