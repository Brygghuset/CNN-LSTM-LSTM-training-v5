# AWS Configuration Setup Guide

Denna guide visar hur du konfigurerar AWS-miljön för Master POC CNN-LSTM-LSTM v5.0 preprocessing pipeline.

## 📁 Filer som skapats

- `aws_config.env` - AWS-konfigurationsfil med alla nödvändiga variabler
- `src/aws_config_loader.py` - Python-modul för att ladda konfiguration
- `.gitignore` - Förhindrar att känslig information committas

## 🚀 Snabbstart

### 1. Kopiera konfigurationsfilen

```bash
# Kopiera aws_config.env till .env
cp aws_config.env .env
```

### 2. Lägg till dina AWS credentials

Redigera `.env` filen och lägg till dina credentials:

```bash
# Redigera .env filen
nano .env

# Lägg till dina credentials:
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
```

### 3. Testa konfigurationen

```bash
# Testa AWS configuration loader
python src/aws_config_loader.py

# Testa AWS credentials
aws sts get-caller-identity
```

## 🔧 Användning i Python

### Grundläggande användning

```python
from src.aws_config_loader import get_aws_config, validate_aws_config

# Hämta konfiguration
config = get_aws_config()

# Validera konfiguration
if validate_aws_config():
    print("✅ AWS konfiguration är giltig")
    
    # Hämta specifika värden
    bucket = config.get('S3_PRIMARY_BUCKET')
    region = config.get('AWS_REGION')
    print(f"Bucket: {bucket}, Region: {region}")
```

### Använda konfigurationstyper

```python
# Hämta olika datatyper
instance_count = config.get_int('SAGEMAKER_INSTANCE_COUNT', 6)
use_spot = config.get_bool('SAGEMAKER_USE_SPOT_INSTANCES', True)
normalization_range = config.get_list('MASTER_POC_NORMALIZATION_RANGE')

print(f"Instances: {instance_count}, Spot: {use_spot}")
print(f"Normalization: {normalization_range}")
```

### Hämta kompletta konfigurationer

```python
# Hämta SageMaker konfiguration
sm_config = config.get_sagemaker_config()
print(f"SageMaker: {sm_config['instance_type']} x{sm_config['instance_count']}")

# Hämta S3 konfiguration
s3_config = config.get_s3_config()
print(f"S3 Input: {s3_config['input_path']}")

# Hämta processing konfiguration
proc_config = config.get_processing_config()
print(f"Case Range: {proc_config['case_range']}")
```

## 🔐 Säkerhet

### Viktiga säkerhetsregler

1. **ALDRIG committa `.env` filer** - De innehåller känslig information
2. **Använd `.gitignore`** - Förhindrar oavsiktliga commits
3. **Rotera credentials regelbundet** - Säkerhetspraxis
4. **Använd IAM roller när möjligt** - Bättre än access keys

### Säkerhetschecklista

- [ ] `.env` filen är i `.gitignore`
- [ ] AWS credentials är korrekta
- [ ] IAM permissions är minimala
- [ ] S3 buckets har rätt access policies
- [ ] SageMaker role har rätt permissions

## 📋 Konfigurationsvariabler

### AWS Account Information
```bash
AWS_ACCOUNT_ID=631158448906
AWS_REGION=eu-north-1
AWS_SAGEMAKER_ROLE_ARN=arn:aws:iam::...
```

### S3 Configuration
```bash
S3_PRIMARY_BUCKET=master-poc-v1.0
S3_INPUT_PATH=s3://master-poc-v1.0/raw-data/
S3_OUTPUT_PATH=s3://master-poc-v1.0/processed-data/
S3_CHECKPOINT_PATH=s3://master-poc-v1.0/checkpoints/
```

### SageMaker Configuration
```bash
SAGEMAKER_INSTANCE_TYPE=ml.m5.2xlarge
SAGEMAKER_INSTANCE_COUNT=6
SAGEMAKER_USE_SPOT_INSTANCES=true
SAGEMAKER_MAX_RUN_TIME=93600
```

### Processing Configuration
```bash
PROCESSING_CASE_RANGE=1-3000
PROCESSING_BATCH_SIZE=50
PROCESSING_CHECKPOINT_INTERVAL=50
MASTER_POC_TIMESERIES_FEATURES=16
MASTER_POC_STATIC_FEATURES=6
MASTER_POC_OUTPUT_FEATURES=8
```

## 🧪 Testning

### Testa AWS integration

```python
# Testa S3 access
from src.aws_config_loader import get_aws_config
import boto3

config = get_aws_config()
s3_client = boto3.client('s3')

# Testa bucket access
bucket = config.get('S3_PRIMARY_BUCKET')
response = s3_client.head_bucket(Bucket=bucket)
print(f"✅ Bucket {bucket} är tillgänglig")
```

### Testa konfigurationsvalidering

```python
from src.aws_config_loader import validate_aws_config, print_aws_config

# Validera konfiguration
if validate_aws_config():
    print("✅ Alla kritiska konfigurationer finns")
    print_aws_config()  # Skriv ut sammanfattning
else:
    print("❌ Konfiguration saknar kritiska värden")
```

## 🔄 Uppdateringar

### Lägga till nya konfigurationer

1. Lägg till variabeln i `aws_config.env`
2. Uppdatera `src/aws_config_loader.py` om nödvändigt
3. Testa med `python src/aws_config_loader.py`

### Ändra befintliga konfigurationer

1. Redigera `.env` filen
2. Testa ändringarna
3. Uppdatera dokumentation om nödvändigt

## 🆘 Felsökning

### Vanliga problem

**Problem**: `aws_config.env inte hittad`
```bash
# Lösning: Kopiera filen
cp aws_config.env .env
```

**Problem**: `AWS credentials invalid`
```bash
# Lösning: Verifiera credentials
aws sts get-caller-identity
```

**Problem**: `S3 bucket access denied`
```bash
# Lösning: Kontrollera IAM permissions
aws s3 ls s3://master-poc-v1.0/
```

### Debugging

```python
# Aktivera debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Testa konfiguration
from src.aws_config_loader import AWSConfig
config = AWSConfig()
config.print_config_summary()
```

## 📚 Ytterligare resurser

- [AWS CLI Configuration](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html)
- [SageMaker IAM Roles](https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html)
- [S3 Bucket Policies](https://docs.aws.amazon.com/AmazonS3/latest/userguide/bucket-policies.html)
- [AWS Security Best Practices](https://aws.amazon.com/security/security-resources/)

---

**Version**: 5.0  
**Skapad**: 2025-10-08  
**Status**: Redo för användning
