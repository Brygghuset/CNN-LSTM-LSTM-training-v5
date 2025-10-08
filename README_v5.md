# Master POC Preprocessing v5.0

**Framgångsrik preprocessing av 3000 cases med full funktionalitet**

## Översikt

Master POC Preprocessing v5.0 är en komplett rewrite av preprocessing-pipeline för CNN-LSTM-LSTM modellen. Denna version implementerar alla kritiska funktioner som saknades i tidigare körningar och säkerställer framgångsrik preprocessing av 3000 cases.

## Kritiska Förbättringar från v4.x

### ✅ **Multi-Instance Case Distribution**
- **Problem**: Alla instanser processade samma cases (6x redundant arbete)
- **Lösning**: Modulo-baserad distribution av cases mellan instanser
- **Resultat**: Ingen dubbel processing, optimal resursutnyttjande

### ✅ **Incremental TFRecord Save**
- **Problem**: Data sparades endast vid completion (ALL data förloras vid timeout)
- **Lösning**: TFRecord sparas löpande per batch
- **Resultat**: Data bevaras även vid timeout/interrupt

### ✅ **Enable Checkpoints Default=True**
- **Problem**: Checkpoints aldrig aktiverade (ingen resume möjlighet)
- **Lösning**: `--enable-checkpoints` satt till `true` som default
- **Resultat**: Automatisk resume vid spot instance restarts

### ✅ **Robust Case Format Parsing**
- **Problem**: Stödde inte blandat format "1-10,17,0022"
- **Lösning**: Utökad parsing som hanterar mixed formats
- **Resultat**: Flexibel case selection

### ✅ **Graceful Shutdown Handling**
- **Problem**: Ingen SIGTERM hantering vid timeout
- **Lösning**: Signal handler för graceful shutdown
- **Resultat**: Data sparas innan MaxRuntimeExceeded

### ✅ **S3 Retry Logic**
- **Problem**: Inga retries vid S3 failures
- **Lösning**: Exponential backoff för alla S3 operations
- **Resultat**: Robusthet mot spot instance restarts

### ✅ **Output Verification**
- **Problem**: False positives i success loggar
- **Lösning**: Verifiering att filer faktiskt skapas
- **Resultat**: Pålitlig success/failure reporting

## Master POC Specifikation

### Input Features
- **Timeseries**: 16 features (7 vital + 3 drugs + 6 ventilator)
- **Static**: 6 features (age, sex, height, weight, bmi, asa)
- **Total Input**: 22 features

### Output Features
- **Drug Predictions**: 3 (Propofol, Remifentanil, Noradrenalin)
- **Ventilator Predictions**: 5 (TV, PEEP, FIO2, RR, etSEV)
- **Total Output**: 8 predictions

### Processing Parameters
- **Window Size**: 300 sekunder (5 min)
- **Step Size**: 30 sekunder (10% overlap)
- **Normalization**: [-1, 1] unified normalization
- **Imputation**: Smart forward fill med clinical zeros

## Mappstruktur

```
CNN-LSTM-LSTM-training-v5/
├── src/
│   ├── master_poc_preprocessing_v5.py          # Ny entry point
│   ├── checkpoint_manager.py                   # Checkpoint system
│   ├── memory_efficient_batch_processor.py   # Memory management
│   └── data/                                   # Data processing modules
├── scripts/
│   └── start_master_poc_v5_preprocessing.py   # Starter script
├── configs/
│   └── master_poc_v5_config.yaml              # Konfiguration
├── tests/                                      # Test suite
│   ├── unit_tests/
│   ├── integration_tests/
│   └── end_to_end_tests/
└── docs/
    ├── AWS_CHECKLIST_V5.0_3000_CASES.md       # Implementation checklist
    └── AWS_TEST-LIST_V5.0_3000_cases.md       # Test suite (132 tests)
```

## Snabbstart

### 1. Småskalig Test (10 cases, 2 instanser)
```bash
python scripts/start_master_poc_v5_preprocessing.py \
    --cases "1-10" \
    --instance-count 2 \
    --instance-type ml.m5.large \
    --max-run 3600
```

### 2. Pilot Test (200 cases, 2 instanser)
```bash
python scripts/start_master_poc_v5_preprocessing.py \
    --cases "1-200" \
    --instance-count 2 \
    --instance-type ml.m5.xlarge \
    --max-run 7200
```

### 3. Production Run (3000 cases, 6 instanser)
```bash
python scripts/start_master_poc_v5_preprocessing.py \
    --cases "1-3000" \
    --instance-count 6 \
    --instance-type ml.m5.2xlarge \
    --use-spot \
    --max-run 93600 \
    --max-wait 187200
```

## Konfiguration

### AWS SageMaker
- **Instance Type**: `ml.m5.2xlarge` (8 vCPU, 32GB RAM)
- **Instance Count**: 6 (parallellisering)
- **Spot Instances**: True (65-70% kostnadsbesparing)
- **Max Run**: 26 timmar
- **Max Wait**: 52 timmar

### S3 Paths
- **Input**: `s3://master-poc-v1.0/raw-data/`
- **Output**: `s3://master-poc-v1.0/processed-data/master-poc-pipeline/`
- **Checkpoints**: `s3://master-poc-v1.0/checkpoints/master-poc-v5-3000cases/`

### Processing
- **Batch Size**: 50 cases per batch
- **Checkpoint Interval**: 50 cases
- **Enable Checkpoints**: True (default)
- **Memory Efficient**: Auto-detect

## Testning

Komplett test suite med 132 tester:

- **Unit Tests (89)**: Testar enskilda funktioner
- **Integration Tests (28)**: Testar samverkan mellan komponenter
- **End-to-End Tests (8)**: Testar hela pipeline
- **Production Tests (6)**: Skarpa AWS-tester
- **Regression Tests (8)**: Förhindrar återfall

Kör tester:
```bash
# Unit tests
python -m pytest tests/unit_tests/

# Integration tests
python -m pytest tests/integration_tests/

# End-to-end tests
python -m pytest tests/end_to_end_tests/
```

## Verifiering

### Efter Körning
1. **S3 Output**: Kontrollera att TFRecord-filer skapades
2. **Metadata**: Verifiera preprocessing_metadata.json
3. **Logs**: Kontrollera CloudWatch logs för errors
4. **Cost**: Verifiera kostnadsbesparing med spot instances

### Förväntade Resultat
- **TFRecord-filer**: train.tfrecord, validation.tfrecord, test.tfrecord
- **Metadata**: Total samples, split counts, shapes
- **Success Rate**: >95%
- **Cost**: ~$12-15 med spot instances

## Troubleshooting

### Vanliga Problem
1. **No TFRecord Output**: Kontrollera att incremental save fungerar
2. **Double Processing**: Verifiera case distribution logic
3. **Checkpoint Failures**: Kontrollera S3 permissions
4. **Timeout**: Överväg att öka max_run eller minska case count

### Debug Commands
```bash
# Lista S3 output
aws s3 ls s3://master-poc-v1.0/processed-data/master-poc-pipeline/ --recursive

# Kontrollera checkpoints
aws s3 ls s3://master-poc-v1.0/checkpoints/master-poc-v5-3000cases/ --recursive

# CloudWatch logs
aws logs describe-log-groups --log-group-name-prefix /aws/sagemaker/
```

## Version History

- **v5.0**: Komplett rewrite med alla kritiska funktioner
- **v4.1**: Fungerande för 2051 cases (men saknade vissa funktioner)
- **v4.0**: Misslyckad 3000-case körning (dubbel processing, ingen output)

## Support

För frågor eller problem, se:
- `docs/AWS_CHECKLIST_V5.0_3000_CASES.md` - Implementation checklist
- `docs/AWS_TEST-LIST_V5.0_3000_cases.md` - Test suite
- CloudWatch logs för runtime debugging

---

**Version**: 5.0.0  
**Skapad**: 2025-10-07  
**Status**: Redo för production  
**Next Review**: Efter pilot 200 cases

