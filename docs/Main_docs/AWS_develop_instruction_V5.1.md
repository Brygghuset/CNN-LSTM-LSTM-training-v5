# AWS Development Instruction v5.0 - Master POC Preprocessing

**Målgrupp**: Nybörjare som använder Cursor AI  
**Syfte**: Utveckla preprocessing för 3000 VitalDB cases på AWS SageMaker  
**Kompletterande dokument**: `AWS_CHECKLIST_V5.0_3000_CASES.md`, `Master POC CNN-LSTM-LSTM.md`

---

## 1. ÖVERSIKT

### 1.1 Vad Ska Byggas
En AWS SageMaker preprocessing pipeline som:
- Läser 3000 VitalDB `.vital` filer från S3
- Processar fysiologisk tidsseriedata enligt Master POC specifikation
- Sparar TFRecord-filer för CNN-LSTM-LSTM modellträning
- Hanterar spot instances, checkpoints och multi-instance distribution

### 1.2 Framgångsrik Referens
**2051-case körning (Juli 2025)** fungerade med:
- Entry point: `pytorch_preprocessing_entry_point.py`
- Guide: AWS SageMaker Preprocessing Guide v4.1
- Single instance (ml.m5.xlarge)
- On-demand (inte spot instances)
- Output: TFRecord-filer med train/val/test split

### 1.3 Kritiska Förbättringar i v5.0
- ✅ Multi-instance case distribution (6 instanser)
- ✅ Spot instances för kostnadsbesparing (~70%)
- ✅ Incremental TFRecord save (data bevaras vid timeout)
- ✅ Checkpoints aktiverade som default
- ✅ Output verification (kontrollera att filer skapas)

---

## 2. AWS MILJÖ SETUP

### 2.1 AWS Konto Information
```yaml
# VIKTIGT: Känslig information finns i aws_config.env filen
AWS Account ID: [Se AWS_ACCOUNT_ID i aws_config.env]
Region: eu-north-1 (Stockholm)
IAM Role: [Se AWS_SAGEMAKER_ROLE_ARN i aws_config.env]
```

**Anmärkning**: Alla känsliga AWS-konfigurationer finns i `aws_config.env` filen. 
Se denna fil för aktuella värden och instruktioner om hur du konfigurerar din miljö.

### 2.2 S3 Buckets
**Primär bucket**: [Se `S3_PRIMARY_BUCKET` i `aws_config.env`]
```
s3://[S3_PRIMARY_BUCKET]/
├── raw-data/                    # VitalDB input data
│   ├── 0001.vital
│   ├── 0002.vital
│   └── ... (3000 filer)
├── processed-data/              # TFRecord output
│   └── master-poc-pipeline/
│       ├── train.tfrecord
│       ├── validation.tfrecord
│       └── test.tfrecord
└── checkpoints/                 # Checkpoint storage
    └── master-poc-v5-3000cases/
```

**Sekundär bucket**: [Se `S3_SECONDARY_BUCKET` i `aws_config.env`]
- Används för SageMaker artifacts och logs

### 2.3 AWS CLI Setup
```bash
# Installera AWS CLI (om inte redan installerat)
pip install awscli

# Konfigurera credentials
aws configure
# AWS Access Key ID: [dina credentials]
# AWS Secret Access Key: [dina credentials]
# Default region: eu-north-1
# Default output format: json

# Verifiera access
aws s3 ls s3://[S3_PRIMARY_BUCKET]/
```

**Anmärkning**: Se `aws_config.env` för aktuella bucket-namn och S3-paths.

---

## 3. PYTHON MILJÖ & DEPENDENCIES

### 3.1 Bibliotek och Versioner
**Från lyckad 2051-case körning**:

```txt
# requirements.txt
tensorflow==2.12.0          # Neural network preprocessing
vitaldb==1.5.6              # VitalDB .vital fil parsing
pyyaml==6.0                 # Konfigurationshantering
scikit-learn==1.2.2         # Train/val/test split
numpy>=1.22.0               # Numeriska operationer
pandas>=1.5.0               # Datahantering
boto3>=1.26.0               # AWS S3 integration
psutil==5.9.0               # Memory monitoring
```

### 3.2 Framework Konfiguration
```yaml
Framework: PyTorch Estimator (SageMaker)
PyTorch Version: 1.12.1
Python Version: py38 (Python 3.8)
Backend: TensorFlow 2.12.0 (för neural preprocessing)
```

**Varför PyTorch Estimator?**
- Automatisk kod-uppladdning (ingen Docker build)
- TensorFlow inkluderat för neural network preprocessing
- Snabb iteration under utveckling
- AWS-managed environment

### 3.3 Verifiera Dependencies

Följande dependencies måste finnas för Master POC v5.0:

**Core Dependencies (från requirements.txt):**
- tensorflow==2.12.0
- vitaldb==1.5.6
- pyyaml==6.0
- scikit-learn==1.2.2
- numpy>=1.22.0
- pandas>=1.5.0
- boto3>=1.26.0
- psutil==5.9.0

**Verifiera att alla moduler finns:**
```bash
# Från project root
python -c "from src.checkpoint_manager import MasterPOCCheckpointManager; print('✅ checkpoint_manager OK')"
python -c "from src.memory_efficient_batch_processor import MemoryEfficientBatchProcessor; print('✅ memory_efficient_batch_processor OK')"
python -c "from src.data.master_poc_preprocessing_orchestrator import MasterPOCPreprocessingOrchestrator; print('✅ orchestrator OK')"
python -c "from src.data.master_poc_preprocessing_pipeline import MasterPOCPreprocessingPipeline; print('✅ pipeline OK')"
python -c "from src.data.master_poc_tfrecord_creator import MasterPOCTFRecordCreator; print('✅ tfrecord_creator OK')"
python -c "from src.utils.case_range_parser import parse_case_range; print('✅ case_range_parser OK')"
```

**Om någon import misslyckas:**
1. Kontrollera att filen finns i `src/` strukturen
2. Kontrollera att `__init__.py` finns i alla directories
3. Kontrollera att PYTHONPATH inkluderar project root

---

## 4. MASTER POC SPECIFIKATION

### 4.1 Input Features (22 totalt)

#### Timeseries Features (16) (se separat dokument)
|
#### Static Features (6) (se separat dokument)

### 4.2 Output Features (8) (se separat dokument)


### 4.3 Processing Parameters
```yaml
Window Size: 300 sekunder (5 minuter)
Step Size: 30 sekunder (10% overlap)
Normalization Range: [-1, 1]
Imputation Method: Smart Forward Fill med clinical zeros
Train/Val/Test Split: 70% / 15% / 15%
```

### 4.4 Unified Normalization Formula
```python
normalized = (value - min_clinical) / (max_clinical - min_clinical) × 2 - 1
```
- Alla features normaliseras till [-1, 1] range
- Använder clinical min/max från Master POC spec
- Neural network stability och LSTM convergence

---

## 5. PROJEKTSTRUKTUR

### 5.1 Mappstruktur
```
CNN-LSTM-LSTM-training-v5/
├── src/
│   ├── master_poc_preprocessing_v5.py          # Entry point
│   ├── checkpoint_manager.py                   # Checkpoint system
│   ├── memory_efficient_batch_processor.py     # Memory management
│   └── data/
│       ├── master_poc_preprocessing_pipeline.py
│       ├── master_poc_preprocessing_orchestrator.py
│       ├── master_poc_unit_conversion.py
│       ├── master_normalizer.py
│       ├── tfrecord_standard_format_standalone.py
│       └── preprocessing/
│           └── master_poc_smart_forward_fill.py
├── configs/
│   └── master_poc_v5_config.yaml               # Konfiguration
├── scripts/
│   └── start_master_poc_v5_preprocessing.py    # Starter script
├── tests/
│   ├── unit_tests/
│   ├── integration_tests/
│   └── end_to_end_tests/
└── docs/
    ├── AWS_CHECKLIST_V5.0_3000_CASES.md
    ├── AWS_TEST-LIST_V5.0_3000_cases.md
    └── AWS_develop_instruction_V5.0.md
```

### 5.2 Kärnfiler och Deras Syfte

#### `master_poc_preprocessing_v5.py` (Entry Point)
**Syfte**: Huvudskript som SageMaker kör
**Innehåll**:
- Argument parsing (cases, batch-size, etc.)
- Multi-instance case distribution
- VitalDB data loading från S3
- Master POC preprocessing pipeline
- TFRecord creation med train/val/test split
- Checkpoint save/resume logic
- S3 upload och verification

#### `checkpoint_manager.py`
**Syfte**: Hantera checkpoints för resume vid spot instance interrupts
**Innehåll**:
- Save checkpoint state till S3
- Load checkpoint och identifiera remaining cases
- Per-instance checkpoint paths

#### `memory_efficient_batch_processor.py`
**Syfte**: Processar stora datasets utan memory overflow
**Innehåll**:
- Batch processing logic
- Memory monitoring
- Garbage collection

#### `master_poc_preprocessing_pipeline.py`
**Syfte**: Master POC specifik preprocessing logic
**Innehåll**:
- Feature mapping (16 timeseries + 6 static)
- Unit conversions (drugs, tidal volume)
- Smart forward fill imputation
- Unified normalization [-1, 1]
- Window creation (300s, 30s step)

#### `tfrecord_standard_format_standalone.py`
**Syfte**: Skapa TFRecord-filer för CNN-LSTM-LSTM training
**Innehåll**:
- TFRecord writer
- Train/val/test split logic
- Metadata generation

---

## 6.5 Master POC Orchestrator - Arkitektur

### Komponentöversikt

Master POC preprocessing använder en **Orchestrator-pattern** för att separera concerns:

1. **MasterPOCPreprocessingPipeline** (Core Logic)
   - Feature mapping (16 timeseries + 6 static)
   - Unit conversions (drugs, tidal volume)
   - Smart forward fill imputation
   - Unified normalization [-1, 1]
   - Window creation (300s, 30s step)

2. **MasterPOCPreprocessingOrchestrator** (Infrastructure)
   - Batch processing
   - Checkpoint management
   - TFRecord creation
   - S3 integration
   - Memory management
   - Error handling

3. **Entry Point** (`master_poc_preprocessing_v5.py`)
   - Multi-instance case distribution
   - SageMaker environment integration
   - Graceful shutdown handling
   - Output verification

### Dataflöde

```
Case IDs → Orchestrator.load_case_data() → Pipeline.preprocess_case() 
  → Orchestrator.process_case() → TFRecord preparation
  → Batch accumulation → Train/Val/Test split → TFRecord files → S3
```

### Användning i Entry Point

```python
# Skapa orchestrator
orchestrator = create_master_poc_orchestrator(
    s3_bucket='master-poc-v1.0',
    checkpoint_interval=50,
    batch_size=50,
    enable_s3=True
)

# Processa case
result = orchestrator.process_case(case_id, timeseries_df, clinical_df)

# Result innehåller:
# - status: 'success' eller 'failed'
# - tfrecord_data: Preprocessad data redo för TFRecord
# - metadata: Processing metadata
# - master_poc_compliance: Validering mot spec
```

### Fördelar med Orchestrator

- **Separation of Concerns**: Core preprocessing logic separerad från infrastructure
- **Testbarhet**: Pipeline kan testas isolerat
- **Återanvändbarhet**: Pipeline kan användas i andra contexts
- **Robusthet**: Orchestrator hanterar fel, checkpoints, memory
- **Skalbarhet**: Batch processing och streaming TFRecord writing

---

## 6. KONFIGURATIONSFIL

### 6.1 `master_poc_v5_config.yaml` Struktur
```yaml
# Master POC Specification
master_poc:
  timeseries_features: 16
  static_features: 6
  output_features: 8
  window_size: 300
  step_size: 30
  normalization_range: [-1, 1]

# AWS SageMaker Configuration
aws:
  instance_type: "ml.m5.2xlarge"      # 8 vCPU, 32GB RAM
  instance_count: 6                   # Multi-instance
  use_spot_instances: true            # Kostnadsbesparing
  max_run: 93600                      # 26 timmar
  max_wait: 187200                    # 52 timmar
  framework_version: "1.12.1"
  python_version: "py38"

# S3 Configuration
s3:
  bucket: "master-poc-v1.0"
  input_path: "s3://master-poc-v1.0/raw-data/"
  output_path: "s3://master-poc-v1.0/processed-data/master-poc-pipeline/"
  checkpoint_path: "s3://master-poc-v1.0/checkpoints/master-poc-v5-3000cases/"

# Processing Configuration
processing:
  batch_size: 50
  checkpoint_interval: 50
  enable_checkpoints: true            # KRITISKT: Måste vara true
  enable_memory_efficient: "auto"

# Master POC Specific
master_poc_specific:
  imputation_method: "master_poc_smart_forward_fill"
  pipeline_type: "master_poc"
  
  # Clinical zeros för imputation
  clinical_zeros:
    HR: 70
    BP_SYS: 140
    BP_DIA: 80
    BP_MAP: 93
    SPO2: 96
    ETCO2: 0.0
    BIS: 0.0
    Propofol_INF: 0.0
    Remifentanil_INF: 0.0
    Noradrenalin_INF: 0.0
    TV: 0.0
    PEEP: 0.0
    FIO2: 0.0
    RR: 0.0
    etSEV: 0.0
    inSev: 0.0

  # Unit conversions
  unit_conversions:
    propofol_concentration: 20.0      # mg/ml
    remifentanil_concentration: 20.0  # μg/ml
    noradrenalin_concentration: 0.1   # mg/ml

# Validation Configuration
validation:
  min_success_rate: 0.95
  min_window_count: 100000
  verify_output_files: true
  verify_s3_upload: true

# Logging
logging:
  level: "INFO"
  format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "/opt/ml/output/preprocessing.log"
```

### 6.2 Varför YAML för Konfiguration?
- **Läsbarhet**: Tydlig hierarkisk struktur
- **Flexibilitet**: Enkelt att ändra parametrar utan kodändringar
- **Versionskontroll**: Spåra konfigurationsändringar i Git
- **Best Practice**: Standard inom ML-projekt

---

## 7. KRITISKA IMPLEMENTATIONER

### 7.1 Multi-Instance Case Distribution
**Problem**: Alla 6 instanser processar samma 3000 cases = 6x redundant arbete

**Lösning**: Distribuera cases med modulo
```python
# I master_poc_preprocessing_v5.py

def get_sagemaker_host_info():
    """Hämta SageMaker host information."""
    training_env = json.loads(os.environ.get('SM_TRAINING_ENV', '{}'))
    current_host = training_env.get('current_host', 'algo-1')
    all_hosts = training_env.get('hosts', ['algo-1'])
    host_index = all_hosts.index(current_host)
    total_hosts = len(all_hosts)
    
    return {
        'current_host': current_host,
        'host_index': host_index,
        'total_hosts': total_hosts
    }

def distribute_cases_across_instances(case_ids, host_info):
    """Dela cases mellan instanser."""
    host_index = host_info['host_index']
    total_hosts = host_info['total_hosts']
    
    # Instance 0 får case 0, 6, 12, ...
    # Instance 1 får case 1, 7, 13, ...
    instance_cases = [
        case_id for i, case_id in enumerate(case_ids)
        if i % total_hosts == host_index
    ]
    
    return instance_cases
```

**Environment Variables**:
- `SM_TRAINING_ENV`: JSON med host info
- `SM_CURRENT_HOST`: Nuvarande host ID (algo-1, algo-2, etc.)
- `SM_HOSTS`: Lista av alla hosts

### 7.2 Incremental TFRecord Save
**Problem**: Data sparas endast vid completion = ALL data förloras vid timeout

**Lösning**: Spara TFRecord löpande per batch
```python
# Spara efter varje batch
for batch in batches:
    windows, targets, static = process_batch(batch)
    
    # Spara omedelbart till TFRecord
    append_to_tfrecord(
        windows=windows,
        targets=targets,
        static=static,
        output_path=tfrecord_path
    )
    
    # Spara checkpoint
    if batch_num % checkpoint_interval == 0:
        checkpoint_manager.save_checkpoint(
            processed_cases=processed_cases,
            batch_num=batch_num
        )
```

### 7.3 Checkpoint System
**Problem**: Checkpoints aldrig aktiverade = ingen resume möjlighet

**Lösning**: Enable checkpoints som default
```python
# I argument parsing
parser.add_argument(
    '--enable-checkpoints',
    action='store_true',
    default=True,  # KRITISKT: Default True
    help='Enable checkpoint/resume functionality'
)

# I starter script hyperparameters
hyperparameters = {
    'enable-checkpoints': 'true',  # Explicit sätt till true
    ...
}
```

### 7.4 Graceful Shutdown
**Problem**: Ingen SIGTERM hantering = data förloras vid MaxRuntimeExceeded

**Lösning**: Signal handler för graceful shutdown
```python
import signal

shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    logger.info("⚠️ Shutdown signal received, finishing current batch...")
    shutdown_requested = True

# Registrera signal handler
signal.signal(signal.SIGTERM, signal_handler)

# I processing loop
for batch in batches:
    if shutdown_requested:
        logger.info("Graceful shutdown: saving progress...")
        save_checkpoint()
        upload_partial_results()
        break
    
    process_batch(batch)
```

### 7.5 S3 Retry Logic
**Problem**: Inga retries vid S3 failures = spot instance restarts kan orsaka fel

**Lösning**: Exponential backoff för S3 operations
```python
import time

def upload_to_s3_with_retry(local_path, s3_path, max_retries=3):
    for attempt in range(max_retries):
        try:
            s3_client.upload_file(local_path, bucket, s3_path)
            logger.info(f"✅ Uploaded to {s3_path}")
            return True
        except Exception as e:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(f"Retry {attempt+1}/{max_retries} after {wait_time}s")
            time.sleep(wait_time)
    
    logger.error(f"❌ Failed to upload after {max_retries} retries")
    return False
```

### 7.6 Output Verification
**Problem**: False positives i success loggar = ingen validering att filer skapas

**Lösning**: Verifiera S3 output
```python
def verify_output_files(s3_bucket, output_prefix):
    """Verifiera att TFRecord-filer skapades."""
    s3_client = boto3.client('s3')
    
    expected_files = [
        'train.tfrecord',
        'validation.tfrecord',
        'test.tfrecord',
        'preprocessing_metadata.json'
    ]
    
    for file_name in expected_files:
        s3_key = f"{output_prefix}/{file_name}"
        try:
            response = s3_client.head_object(Bucket=s3_bucket, Key=s3_key)
            file_size = response['ContentLength']
            logger.info(f"✅ {file_name}: {file_size/1e6:.1f} MB")
        except:
            logger.error(f"❌ Missing: {file_name}")
            return False
    
    return True
```

---

## 8. SAGEMAKER ESTIMATOR KONFIGURATION

### 8.1 PyTorch Estimator Setup
```python
from sagemaker.pytorch import PyTorch

estimator = PyTorch(
    # Entry point
    entry_point='master_poc_preprocessing_v5.py',
    source_dir='src/',
    
    # AWS Configuration
    role='arn:aws:iam::631158448906:role/service-role/AmazonSageMaker-ExecutionRole-20250714T085596',
    
    # Framework
    framework_version='1.12.1',
    py_version='py38',
    
    # Instance Configuration
    instance_count=6,
    instance_type='ml.m5.2xlarge',
    
    # Spot Instances
    use_spot_instances=True,
    max_run=93600,      # 26 timmar
    max_wait=187200,    # 52 timmar
    
    # Hyperparameters
    hyperparameters={
        'cases': '1-3000',
        'batch-size': 50,
        'checkpoint-interval': 50,
        'enable-checkpoints': 'true',
        'timeseries-features': 16,
        'static-features': 6,
        'output-features': 8,
        'normalization-range': '-1.0 1.0',
        'imputation-method': 'master_poc_smart_forward_fill',
        'pipeline-type': 'master_poc',
        's3-bucket': 'master-poc-v1.0'
    },
    
    # S3 Paths
    output_path='s3://master-poc-v1.0/processed-data/master-poc-pipeline/',
    code_location='s3://master-poc-v1.0/master-poc-code/',
    
    # Job Configuration
    base_job_name='master-poc-preprocessing-v5',
    enable_sagemaker_metrics=True
)
```

### 8.2 Hyperparameters Förklaring
| Parameter | Värde | Syfte |
|-----------|-------|-------|
| `cases` | `1-3000` | Case range att processa |
| `batch-size` | `50` | Cases per batch (memory management) |
| `checkpoint-interval` | `50` | Spara checkpoint var 50:e case |
| `enable-checkpoints` | `true` | Aktivera checkpoint system |
| `timeseries-features` | `16` | Antal timeseries features (Master POC spec) |
| `static-features` | `6` | Antal static features (Master POC spec) |
| `output-features` | `8` | Antal output predictions (Master POC spec) |
| `normalization-range` | `-1.0 1.0` | Normalization range (mellanslag, INTE komma) |
| `imputation-method` | `master_poc_smart_forward_fill` | Imputation strategi |
| `pipeline-type` | `master_poc` | Pipeline identifier |
| `s3-bucket` | `master-poc-v1.0` | S3 bucket för data |

### 8.3 Instance Type Val
**ml.m5.2xlarge**:
- 8 vCPU
- 32 GB RAM
- $0.461/timme (on-demand)
- $0.14-0.16/timme (spot, ~70% besparing)

**Varför denna instance type?**
- Tillräckligt RAM för sliding window processing
- Balans mellan kostnad och prestanda
- Testad i tidigare körningar

---

## 9. UTVECKLINGSWORKFLOW

### 9.1 Steg-för-Steg Process

#### Steg 1: Setup Lokal Miljö
```bash
# Klona repository
git clone <repo-url>
cd CNN-LSTM-LSTM-training-v5

# Skapa virtual environment
python3.8 -m venv venv
source venv/bin/activate

# Installera dependencies
pip install -r requirements.txt

# Konfigurera AWS CLI
aws configure
```

#### Steg 2: Verifiera S3 Access
```bash
# Lista raw data
aws s3 ls s3://master-poc-v1.0/raw-data/ --recursive | head -10

# Kontrollera antal .vital filer
aws s3 ls s3://master-poc-v1.0/raw-data/ --recursive | grep ".vital" | wc -l
```

#### Steg 3: Utveckla Entry Point
```bash
# Öppna entry point i Cursor AI
cursor src/master_poc_preprocessing_v5.py

# Använd Cursor AI för att:
# - Implementera multi-instance distribution
# - Integrera checkpoint manager
# - Lägga till TFRecord creation
# - Implementera graceful shutdown
```

#### Steg 4: Testa Lokalt (Valfritt)
```bash
# Test med 1 case lokalt
python src/master_poc_preprocessing_v5.py \
    --cases "0001" \
    --batch-size 1 \
    --enable-checkpoints \
    --s3-bucket master-poc-v1.0
```

#### Steg 5: Småskalig AWS Test (10 cases, 2 instanser)
```bash
python scripts/start_master_poc_v5_preprocessing.py \
    --cases "1-10" \
    --instance-count 2 \
    --instance-type ml.m5.large \
    --max-run 3600
```

#### Steg 6: Pilot Test (200 cases, 2 instanser)
```bash
python scripts/start_master_poc_v5_preprocessing.py \
    --cases "1-200" \
    --instance-count 2 \
    --instance-type ml.m5.xlarge \
    --max-run 7200
```

#### Steg 7: Production Run (3000 cases, 6 instanser)
```bash
python scripts/start_master_poc_v5_preprocessing.py \
    --cases "1-3000" \
    --instance-count 6 \
    --instance-type ml.m5.2xlarge \
    --use-spot \
    --max-run 93600 \
    --max-wait 187200
```

### 9.2 Monitoring Under Körning
```bash
# Lista SageMaker jobs
aws sagemaker list-training-jobs --max-results 5

# Få job status
aws sagemaker describe-training-job --training-job-name <job-name>

# CloudWatch logs
aws logs tail /aws/sagemaker/TrainingJobs --follow
```

### 9.3 Verifiering Efter Körning
```bash
# Lista output filer
aws s3 ls s3://master-poc-v1.0/processed-data/master-poc-pipeline/ --recursive

# Ladda ner metadata
aws s3 cp s3://master-poc-v1.0/processed-data/master-poc-pipeline/preprocessing_metadata.json .

# Kontrollera metadata
cat preprocessing_metadata.json | jq '.'
```

---

## 10. CURSOR AI TIPS


---

## 11. VANLIGA PROBLEM & LÖSNINGAR

### 11.1 PermissionError vid S3 Access
**Problem**: `botocore.exceptions.NoCredentialsError`

**Lösning**:
```bash
# Verifiera AWS credentials
aws sts get-caller-identity

# Kontrollera IAM role permissions
aws iam get-role --role-name AmazonSageMaker-ExecutionRole-20250714T085596
```

### 11.2 Memory Overflow
**Problem**: `MemoryError: Unable to allocate array`

**Lösning**:
- Minska `batch-size` från 50 till 25
- Använd `enable-memory-efficient: true`
- Öka instance type till ml.m5.4xlarge

### 11.3 Spot Instance Interrupt
**Problem**: Job stoppas mitt i körning

**Lösning**:
- Checkpoints sparas automatiskt var 50:e case
- Resume sker automatiskt vid restart
- Verifiera `enable-checkpoints: true`

### 11.4 No TFRecord Output
**Problem**: Success i loggar men inga filer i S3

**Lösning**:
- Kontrollera att `verify_output_files()` körs
- Verifiera S3 upload permissions
- Kontrollera att incremental save är implementerat

### 11.5 Double Processing
**Problem**: Alla instanser processar samma cases

**Lösning**:
- Verifiera att `distribute_cases_across_instances()` körs
- Kontrollera CloudWatch logs för "This Instance Cases"
- Verifiera att `SM_TRAINING_ENV` läses korrekt

---

## 12. KOSTNADSUPPSKATTNING

### 12.1 Instance Kostnader
**ml.m5.2xlarge**:
- On-demand: $0.461/timme
- Spot: $0.14-0.16/timme (~70% besparing)

**3000 cases körning**:
- 6 instanser × 26 timmar = 156 instance-timmar
- On-demand: 156 × $0.461 = ~$72
- Spot: 156 × $0.15 = ~$23
- **Besparing: ~$49 (68%)**

### 12.2 S3 Storage Kostnader
- Raw data: ~60 GB (3000 × 20 MB)
- Processed data: ~10 GB (TFRecord compressed)
- S3 Standard: $0.023/GB/månad
- **Total: ~$1.60/månad**

### 12.3 Total Uppskattad Kostnad
- **Utveckling & Tester**: ~$10-15 (småskaliga körningar)
- **Production Run**: ~$23 (spot instances)
- **S3 Storage**: ~$2/månad
- **Total första månaden**: ~$35-40

---

## 13. CHECKLISTA FÖR IMPLEMENTATION

### 13.1 Före Utveckling
- [ ] AWS CLI konfigurerat och testat
- [ ] S3 buckets verifierade (master-poc-v1.0)
- [ ] IAM role permissions kontrollerade
- [ ] Python 3.8 environment setup
- [ ] Dependencies installerade (requirements.txt)
- [ ] Referensdokument lästa (AWS_CHECKLIST, Master POC spec)

### 13.2 Under Utveckling
- [ ] Entry point skapad (master_poc_preprocessing_v5.py)
- [ ] Multi-instance distribution implementerad
- [ ] Checkpoint manager integrerad
- [ ] TFRecord creation implementerad
- [ ] Graceful shutdown handler tillagd
- [ ] S3 retry logic implementerad
- [ ] Output verification implementerad
- [ ] Konfigurationsfil skapad (master_poc_v5_config.yaml)
- [ ] Starter script skapad (start_master_poc_v5_preprocessing.py)

### 13.3 Före AWS Deploy
- [ ] Lokal test genomförd (om möjligt)
- [ ] Kod granskad för kritiska funktioner
- [ ] Hyperparameters verifierade mot Master POC spec
- [ ] S3 paths verifierade
- [ ] Checkpoint paths skapade i S3

### 13.4 Efter Småskalig Test (10 cases)
- [ ] Multi-instance distribution fungerar (olika cases per instans)
- [ ] Checkpoints sparas korrekt
- [ ] TFRecord-filer skapas
- [ ] S3 upload fungerar
- [ ] Output verification passerar
- [ ] CloudWatch logs granskas för errors

### 13.5 Efter Pilot Test (200 cases)
- [ ] Success rate >95%
- [ ] TFRecord shapes korrekta ([300, 16] input, [8] output)
- [ ] Train/val/test split korrekt (70/15/15)
- [ ] Metadata innehåller rätt statistik
- [ ] Kostnad inom budget

### 13.6 Före Production Run (3000 cases)
- [ ] Alla tidigare tester passerade
- [ ] Spot instance konfiguration verifierad
- [ ] Max run/wait times satta (26h/52h)
- [ ] Monitoring setup (CloudWatch)
- [ ] Budget alert konfigurerad (<$30)

---

## 14. SUPPORT & RESURSER

### 14.1 Dokumentation
- **AWS SageMaker Preprocessing Guide v4.1**: Detaljerad guide för 2051-case körning
- **AWS_CHECKLIST_V5.0_3000_CASES.md**: Implementation checklist med alla kritiska funktioner
- **Master POC CNN-LSTM-LSTM.md**: Exakt specifikation för features och normalization
- **AWS_TEST-LIST_V5.0_3000_cases.md**: 132 tester för verifiering

### 14.2 AWS Dokumentation
- SageMaker PyTorch Estimator: https://sagemaker.readthedocs.io/en/stable/frameworks/pytorch/
- S3 boto3 API: https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html
- CloudWatch Logs: https://docs.aws.amazon.com/AmazonCloudWatch/latest/logs/

### 14.3 VitalDB Resurser
- VitalDB Python Library: https://github.com/vitaldb/vitaldb
- VitalDB Documentation: https://vitaldb.net/dataset/

### 14.4 Cursor AI Resurser
- Cursor Documentation: https://docs.cursor.sh/
- Cursor AI Best Practices: Använd @ för att referera filer, skriv tydliga prompts

---

## 15. SAMMANFATTNING

### 15.1 Vad Du Ska Bygga
En robust AWS SageMaker preprocessing pipeline som:
1. Läser 3000 VitalDB cases från S3
2. Distribuerar cases mellan 6 instanser
3. Processar enligt Master POC spec (16+6 features → 8 outputs)
4. Sparar TFRecord-filer löpande
5. Hanterar spot instance interrupts med checkpoints
6. Verifierar output och laddar upp till S3

### 15.2 Kritiska Framgångsfaktorer
1. **Multi-Instance Distribution**: Ingen dubbel processing
2. **Incremental Save**: Data bevaras vid timeout
3. **Checkpoints Enabled**: Resume vid spot interrupts
4. **Output Verification**: Verifiera att filer faktiskt skapas
5. **Graceful Shutdown**: Spara progress vid SIGTERM
6. **S3 Retry Logic**: Robusthet mot spot restarts

### 15.3 Förväntade Resultat
- **Output**: 3 TFRecord-filer (train, validation, test)
- **Success Rate**: >95%
- **Processing Time**: ~26 timmar
- **Cost**: ~$23 med spot instances
- **Windows**: >100,000 sliding windows
- **Shapes**: Input [300, 16], Static [6], Output [8]

### 15.4 Nästa Steg
1. Setup AWS miljö och lokal development environment
2. Implementera entry point med alla kritiska funktioner
3. Testa småskaligt (10 cases, 2 instanser)
4. Pilot test (200 cases)
5. Production run (3000 cases)
6. Verifiera output och använd för modellträning

---

**Version**: 5.0  
**Skapad**: 2025-10-08  
**Status**: Redo för implementation  
**Estimerad utvecklingstid**: 2-3 dagar  
**Estimerad kostnad**: $35-40 (inklusive tester)

**Lycka till med utvecklingen! 🚀**
