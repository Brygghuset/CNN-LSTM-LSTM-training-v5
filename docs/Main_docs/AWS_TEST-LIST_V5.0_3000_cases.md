# AWS Test List v5.0 - 3000 Cases Implementation

**Mål**: Komplett testning av alla funktioner för framgångsrik 3000-case preprocessing  
**Baserad på**: AWS_CHECKLIST_V5.0_3000_CASES.md och Master POC CNN-LSTM-LSTM.md  
**Skapad**: 2025-10-07

---

## CORE PROCESSING TESTS

### Case Range Parsing
- [x] T001, Test Basic Range Parsing, Unit Test, Verifiera att "1-100" parsas korrekt till lista med 100 cases
- [x] T002, Test Comma-Separated Parsing, Unit Test, Verifiera att "1,5,10" parsas korrekt till lista med 3 cases
- [x] T003, Test Mixed Format Parsing, Unit Test, Verifiera att "1-10,17,0022" parsas korrekt till lista med 12 cases
- [x] T004, Test Zero-Padded Cases, Unit Test, Verifiera att "0001,0022" hanteras korrekt med zero-padding
- [x] T005, Test Invalid Range Format, Unit Test, Verifiera att ogiltiga format ger tydliga felmeddelanden
- [x] T006, Test Large Range Parsing, Unit Test, Verifiera att "1-3000" parsas effektivt utan minnesöverskridning

### Master POC Feature Mapping
- [x] T007, Test 16 Timeseries Features, Unit Test, Verifiera att exakt 16 timeseries features mappas enligt Master POC spec
- [x] T008, Test 6 Static Features, Unit Test, Verifiera att age, sex, height, weight, bmi, asa mappas korrekt
- [x] T009, Test Feature Order Consistency, Unit Test, Verifiera att feature-ordning följer Master POC spec (HR, BP_SYS, etc.)
- [x] T010, Test Missing Feature Handling, Unit Test, Verifiera hantering när required features saknas i rådata
- [x] T011, Test Feature Name Standardization, Unit Test, Verifiera att alla feature-namn följer standardiserat format

### Unit Conversion
- [x] T012, Test Propofol Conversion, Unit Test, Verifiera konvertering från mL/h till mg/kg/h med 20mg/ml koncentration
- [x] T013, Test Remifentanil Conversion, Unit Test, Verifiera konvertering från mL/h till μg/kg/min med 20μg/ml koncentration
- [x] T014, Test IBW Calculation, Unit Test, Verifiera Devine formula för Ideal Body Weight beräkning
- [x] T015, Test Tidal Volume Conversion, Unit Test, Verifiera konvertering till ml/kg IBW
- [x] T016, Test Pressure Unit Conversion, Unit Test, Verifiera konvertering mellan kPa och cmH2O
- [x] T017, Test Edge Case Weights, Unit Test, Verifiera unit conversion med extrema vikter (20kg, 200kg)

### Smart Forward Fill & Imputation
- [x] T018, Test Clinical Zeros, Unit Test, Verifiera att kliniska nollor appliceras korrekt per parameter
- [x] T019, Test Forward Fill Logic, Unit Test, Verifiera forward fill för isolerade NaN-värden
- [x] T020, Test Backward Fill Fallback, Unit Test, Verifiera backward fill som fallback för initiala NaN
- [x] T021, Test Mean Imputation, Unit Test, Verifiera mean-imputation för isolerade NaN mellan giltiga värden
- [x] T022, Test Default Values, Unit Test, Verifiera default-värden för vital signs (HR=70, BP_SYS=140, etc.)
- [x] T023, Test Imputation Edge Cases, Unit Test, Verifiera hantering av helt tomma serier

### Unified Normalization
- [x] T024, Test Normalization Formula, Unit Test, Verifiera unified normalization formula: (value - min) / (max - min) × 2 - 1
- [x] T025, Test Range [-1, 1], Unit Test, Verifiera att alla normaliserade värden ligger inom [-1, 1]
- [x] T026, Test Clinical Min/Max Ranges, Unit Test, Verifiera att clinical ranges följer Master POC spec
- [x] T027, Test Static Feature Normalization, Unit Test, Verifiera normalization av age, height, weight, bmi, asa
- [x] T028, Test Sex Encoding, Unit Test, Verifiera att sex kodas som -1 (Female) och 1 (Male)
- [x] T029, Test Reverse Normalization, Unit Test, Verifiera att reverse normalization ger ursprungliga värden

### Window Creation
- [x] T030, Test Window Size 300s, Unit Test, Verifiera att windows är exakt 300 sekunder
- [x] T031, Test Step Size 30s, Unit Test, Verifiera att step size är 30 sekunder (10% overlap)
- [x] T032, Test Sliding Window Logic, Unit Test, Verifiera korrekt sliding window implementation
- [x] T033, Test Window Shape [300, 16], Unit Test, Verifiera att window shape är [300, 16] för timeseries
- [x] T034, Test Insufficient Data Handling, Unit Test, Verifiera hantering när case har <300s data
- [x] T035, Test Window Count Calculation, Unit Test, Verifiera korrekt beräkning av förväntat antal windows

---

## OUTPUT & STORAGE TESTS

### TFRecord Creation
- [x] T036, Test TFRecord Schema, Unit Test, Verifiera korrekt TFRecord schema med timeseries, static, targets
- [x] T037, Test Memory-Efficient Streaming, Unit Test, Verifiera att TFRecord skrivs streamat utan minnesöverskridning
- [x] T038, Test Target Shape [8], Unit Test, Verifiera att targets har shape [8] med 3 drugs + 5 ventilator
- [x] T039, Test Static Shape [6], Unit Test, Verifiera att static features har shape [6]
- [x] T040, Test TFRecord File Creation, Integration Test, Verifiera att TFRecord-filer faktiskt skapas på disk
- [x] T041, Test TFRecord Readability, Integration Test, Verifiera att skapade TFRecord-filer kan läsas av TensorFlow

### Train/Val/Test Split
- [x] T042, Test 70/15/15 Split, Unit Test, Verifiera korrekt 70/15/15 split av windows
- [x] T043, Test Split Consistency, Unit Test, Verifiera att samma case alltid hamnar i samma split
- [x] T044, Test Three TFRecord Files, Integration Test, Verifiera att train.tfrecord, validation.tfrecord, test.tfrecord skapas
- [x] T045, Test Split Metadata, Unit Test, Verifiera att split-statistik sparas i metadata
- [x] T046, Test Minimum Split Size, Unit Test, Verifiera hantering när dataset är för litet för split
- [x] T047, Test Split Randomization, Unit Test, Verifiera att split är randomiserad men deterministisk

### Incremental TFRecord Save
- [x] T048, Test Batch-wise Save, Integration Test, Verifiera att TFRecord sparas per batch, inte endast vid completion
- [x] T049, Test Save Frequency, Integration Test, Verifiera att save sker med konfigurerad frekvens
- [x] T050, Test Partial Save Recovery, Integration Test, Verifiera att partiellt sparad data kan återhämtas
- [x] T051, Test Writer Flush, Unit Test, Verifiera att TFRecord writers flushas regelbundet
- [x] T052, Test Save Progress Tracking, Unit Test, Verifiera att save-progress trackas korrekt

### S3 Upload & Metadata
- [x] T053, Test S3 Upload Success, Integration Test, Verifiera att TFRecord-filer laddas upp till S3
- [x] T054, Test Metadata JSON Creation, Unit Test, Verifiera att preprocessing_metadata.json skapas korrekt
- [x] T055, Test Metadata Content, Unit Test, Verifiera att metadata innehåller total_samples, split counts, shapes
- [x] T056, Test S3 Path Structure, Integration Test, Verifiera korrekt S3 path-struktur för output
- [x] T057, Test Upload Before Timeout, Integration Test, Verifiera att upload sker innan MaxRuntimeExceeded

---

## ROBUSTNESS TESTS

### Checkpoint System
- [x] T058, Test Checkpoint Creation, Unit Test, Verifiera att checkpoints skapas med korrekt format
- [x] T059, Test Checkpoint Resume, Integration Test, Verifiera att processing kan återupptas från checkpoint
- [x] T060, Test Checkpoint Interval, Unit Test, Verifiera att checkpoints sparas med konfigurerad interval
- [x] T061, Test Enable Checkpoints Default, Unit Test, Verifiera att --enable-checkpoints är True by default
- [x] T062, Test Checkpoint State Tracking, Unit Test, Verifiera att processed_cases och failed_cases trackas
- [x] T063, Test Checkpoint S3 Upload, Integration Test, Verifiera att checkpoints laddas upp till S3

### Error Handling & Memory Management
- [x] T064, Test Per-Case Error Handling, Unit Test, Verifiera att fel i enskilda cases inte stoppar processing
- [x] T065, Test Error Logging, Unit Test, Verifiera att fel loggas med tillräcklig detail
- [x] T066, Test Memory-Efficient Processing, Integration Test, Verifiera att minnesanvändning hålls under kontroll
- [x] T067, Test Batch Size Optimization, Integration Test, Verifiera att batch size balanserar minne och prestanda
- [x] T068, Test Failed Cases Tracking, Unit Test, Verifiera att failed cases trackas och rapporteras
- [x] T069, Test Graceful Degradation, Integration Test, Verifiera att systemet fortsätter vid partiella fel

---

## MULTI-INSTANCE TESTS

### Case Distribution
- [x] T070, Test SageMaker Host Detection, Unit Test, Verifiera att SM_CURRENT_HOST och SM_HOSTS läses korrekt
- [x] T071, Test Case Partitioning Logic, Unit Test, Verifiera modulo-baserad case distribution
- [ ] T072, Test No Case Overlap, Integration Test, Verifiera att instanser inte processar samma cases
- [ ] T073, Test Even Distribution, Unit Test, Verifiera jämn fördelning av cases mellan instanser
- [ ] T074, Test Host Index Calculation, Unit Test, Verifiera korrekt beräkning av host_index
- [ ] T075, Test Distribution with 6 Instances, Integration Test, Verifiera distribution med 6 instanser

### Distributed Checkpoints
- [x] T076, Test Per-Instance Checkpoint Paths, Unit Test, Verifiera unika checkpoint paths per instans
- [ ] T077, Test Distributed Checkpoint Creation, Integration Test, Verifiera att varje instans skapar sina checkpoints
- [ ] T078, Test Checkpoint Path Conflicts, Integration Test, Verifiera att instanser inte skriver till samma checkpoint
- [ ] T079, Test Instance-Specific Recovery, Integration Test, Verifiera att instanser kan återhämta sina egna checkpoints

---

## SPOT INSTANCE TESTS

### Spot Instance Support
- [ ] T080, Test Spot Instance Configuration, Unit Test, Verifiera att use_spot_instances=True konfigureras korrekt
- [ ] T081, Test Max Wait Time, Unit Test, Verifiera att max_wait = max_run × 2
- [ ] T082, Test Spot Interrupt Simulation, Integration Test, Simulera spot instance interrupt och verifiera recovery
- [ ] T083, Test Checkpoint Resume After Interrupt, Integration Test, Verifiera att processing återupptas efter spot interrupt
- [ ] T084, Test Cost Savings Calculation, Integration Test, Verifiera att spot instances ger förväntad kostnadsbesparing

---

## VALIDATION & VERIFICATION TESTS

### Output Verification
- [ ] T085, Test TFRecord File Existence, Integration Test, Verifiera att TFRecord-filer faktiskt existerar i S3
- [ ] T086, Test TFRecord File Size, Integration Test, Verifiera att TFRecord-filer har rimlig storlek (>100MB)
- [ ] T087, Test S3 List Verification, Integration Test, Verifiera att S3 list kommando returnerar förväntade filer
- [ ] T088, Test File Count Verification, Integration Test, Verifiera att rätt antal filer skapas
- [ ] T089, Test Output Path Structure, Integration Test, Verifiera korrekt S3 output path struktur

### Feature & Window Validation
- [ ] T090, Test 16+6 Input Features, Integration Test, Verifiera att input har exakt 16 timeseries + 6 static features
- [ ] T091, Test 8 Output Features, Integration Test, Verifiera att output har exakt 8 predictions
- [ ] T092, Test Window Count Validation, Integration Test, Verifiera att faktiskt antal windows matchar förväntat
- [ ] T093, Test Success Rate Validation, Integration Test, Verifiera att success rate är >95%
- [ ] T094, Test Feature Range Validation, Integration Test, Verifiera att alla features ligger inom förväntade ranges

---

## NEW CRITICAL FUNCTIONALITY TESTS

### Robust Case Format Parsing
- [ ] T095, Test Mixed Format "1-10,17,0022", Unit Test, Verifiera att blandat format parsas korrekt
- [ ] T096, Test Complex Mixed Format, Unit Test, Verifiera parsing av "1-5,10-15,20,25,30-35"
- [ ] T097, Test Mixed Format Edge Cases, Unit Test, Verifiera hantering av "1-1,5,10-10" (single case ranges)

### Graceful Shutdown Handling
- [ ] T098, Test SIGTERM Signal Handler, Unit Test, Verifiera att SIGTERM signal fångas korrekt
- [ ] T099, Test Graceful Shutdown Logic, Integration Test, Verifiera att pågående processing avslutas gracefully
- [ ] T100, Test Data Save Before Shutdown, Integration Test, Verifiera att data sparas innan shutdown
- [ ] T101, Test Shutdown Timeout Handling, Integration Test, Verifiera hantering av shutdown timeout

### S3 Retry Logic
- [ ] T102, Test S3 Exponential Backoff, Unit Test, Verifiera exponential backoff för S3 retries
- [ ] T103, Test S3 Upload Retry, Integration Test, Verifiera retry vid S3 upload failures
- [ ] T104, Test S3 Download Retry, Integration Test, Verifiera retry vid S3 download failures
- [ ] T105, Test Max Retry Limit, Unit Test, Verifiera att max retry limit respekteras
- [ ] T106, Test Retry Success Recovery, Integration Test, Verifiera att processing fortsätter efter lyckad retry

### Idempotent S3 Writes
- [ ] T107, Test Overwrite Policy, Integration Test, Verifiera att återkörningar skriver över tidigare filer
- [ ] T108, Test Unique Job ID Suffix, Unit Test, Verifiera att unique job_id används för att undvika konflikter
- [ ] T109, Test Multiple Run Handling, Integration Test, Verifiera hantering av multipla körningar av samma jobb
- [ ] T110, Test Idempotent Metadata, Integration Test, Verifiera att metadata skrivs idempotent

---

## END-TO-END INTEGRATION TESTS

### Small Scale Tests (Development)
- [ ] T111, Test 10 Cases 2 Instances, End-to-End Test, Verifiera komplett pipeline med 10 cases och 2 instanser
- [ ] T112, Test Case Distribution E2E, End-to-End Test, Verifiera att case distribution fungerar end-to-end
- [ ] T113, Test Checkpoint Resume E2E, End-to-End Test, Verifiera checkpoint resume i full pipeline
- [ ] T114, Test Incremental Save E2E, End-to-End Test, Verifiera incremental save i full pipeline

### Medium Scale Tests (Pilot)
- [ ] T115, Test 200 Cases 6 Instances, End-to-End Test, Pilot test med 200 cases och 6 instanser
- [ ] T116, Test Output Verification E2E, End-to-End Test, Verifiera all output i pilot test
- [ ] T117, Test Metadata Analysis E2E, End-to-End Test, Analysera metadata från pilot test
- [ ] T118, Test Performance Metrics E2E, End-to-End Test, Mät prestanda i pilot test

### Production Scale Tests (AWS)
- [ ] T119, Test 3000 Cases 6 Instances, Production Test, Full 3000-case test med 6 instanser
- [ ] T120, Test Spot Instance Production, Production Test, 3000-case test med spot instances
- [ ] T121, Test CloudWatch Monitoring, Production Test, Verifiera CloudWatch monitoring under production
- [ ] T122, Test Cost Optimization, Production Test, Verifiera kostnadsbesparing med spot instances
- [ ] T123, Test 26h Runtime Limit, Production Test, Verifiera att 26h runtime är tillräckligt
- [ ] T124, Test Final Output Validation, Production Test, Komplett validering av 3000-case output

---

## REGRESSION TESTS

### Prevent Previous Failures
- [ ] T125, Test No Double Processing, Regression Test, Verifiera att dubbel processing inte sker (från 3000-case failure)
- [ ] T126, Test TFRecord Output Exists, Regression Test, Verifiera att TFRecord-filer faktiskt skapas (från 200/20-case issue)
- [ ] T127, Test Checkpoints Actually Work, Regression Test, Verifiera att checkpoints faktiskt fungerar
- [ ] T128, Test No False Success Logs, Regression Test, Verifiera att success logs matchar faktisk output

### Master POC Spec Compliance
- [ ] T129, Test 16 Not 14 Features, Regression Test, Verifiera att 16 (inte 14) timeseries features används
- [ ] T130, Test 8 Not 7 Outputs, Regression Test, Verifiera att 8 (inte 7) output features används
- [ ] T131, Test Correct Normalization Range, Regression Test, Verifiera [-1, 1] normalization (inte andra ranges)
- [ ] T132, Test Master POC Unit Conversions, Regression Test, Verifiera Master POC-specifika unit conversions

---

**Total Tests**: 132  
**✅ Klara Tests**: 19 (14%)  
**❌ Saknade Tests**: 113 (86%)  

**Klara Tests per kategori**:
- ✅ Case Range Parsing: 4/6 (67%)
- ✅ Master POC Feature Mapping: 5/5 (100%)
- ✅ Unit Conversion: 6/6 (100%)
- ✅ Checkpoint System: 3/6 (50%)
- ✅ Multi-Instance: 3/9 (33%)

**Estimerad Testtid för återstående**:
- Unit Tests: ~15 timmar (19 klara)
- Integration Tests: ~15 timmar  
- End-to-End Tests: ~10 timmar
- Production Tests: ~30 timmar (inkl. väntetid)
- Regression Tests: ~5 timmar

**Total återstående**: ~75 timmar testning

---

**Version**: 5.0  
**Skapad**: 2025-10-07  
**Status**: Redo för implementation  
**Next Review**: Efter implementation av master_poc_preprocessing_v5.py
