# 更改日志

# 发布 1.9 - 10/22/2020

## 主要更新

### 神经网络架构搜索

* Support regularized evolution algorithm for NAS scenario (#2802)
* Add NASBench201 in search space zoo (#2766)

### 模型压缩

* AMC pruner improvement: support resnet, support reproduction of the experiments (default parameters in our example code) in AMC paper (#2876 #2906)
* Support constraint-aware on some of our pruners to improve model compression efficiency (#2657)
* Support "tf.keras.Sequential" in model compression for TensorFlow (#2887)
* Support customized op in the model flops counter (#2795)
* Support quantizing bias in QAT quantizer (#2914)

### 训练平台

* Support configuring python environment using "preCommand" in remote mode (#2875)
* Support AML training service in Windows (#2882)
* Support reuse mode for remote training service (#2923)

### Web 界面和 nnictl

* The "Overview" page on WebUI is redesigned with new layout (#2914)
* Upgraded node, yarn and FabricUI, and enabled Eslint (#2894 #2873 #2744)
* Add/Remove columns in hyper-parameter chart and trials table in "Trials detail" page (#2900)
* JSON format utility beautify on WebUI (#2863)
* Support nnictl command auto-completion (#2857)

## UT & IT

* Add integration test for experiment import and export (#2878)
* Add integration test for user installed builtin tuner (#2859)
* Add unit test for nnictl (#2912)

## 文档

* Refactor of the document for model compression (#2919)

## 修复的 Bug

* Bug fix of naïve evolution tuner, correctly deal with trial fails (#2695)
* Resolve the warning "WARNING (nni.protocol) IPC pipeline not exists, maybe you are importing tuner/assessor from trial code?" (#2864)
* Fix search space issue in experiment save/load (#2886)
* Fix bug in experiment import data (#2878)
* Fix annotation in remote mode (python 3.8 ast update issue) (#2881)
* Support boolean type for "choice" hyper-parameter when customizing trial configuration on WebUI (#3003)

# 发布 1.8 - 8/27/2020

## 主要更新

### 训练平台

* 在 Web 界面直接访问 Trial 日志 (仅支持本地模式) (#2718)
* 添加 OpenPAI Trial Job 详情链接 (#2703)
* 在可重用的环境中支持 GPU 调度器 (#2627) (#2769)
* 为在 `trial_runner` 中的 `web_channel` 添加超时时间 (#2710)
* 在 AzureML 模式下展示环境配置错误信息 (#2724)
* 为在 OpenPAI 模式复制数据增加更多日志信息 (#2702)

### Web 界面，nnictl 和 nnicli

* 改进超参数并行坐标图的绘制 (#2691) (#2759)
* 为 Trial Job 列表添加分页 (#2738) (#2773)
* 使面板可以在鼠标点击其它区域时关闭 (#2734)
* 从 Web 界面中去掉多阶段支持 (#2760)
* 支持保存和加载 Experiment (#2750)
* 在导出结果的命令中增加导出中间结果的选项 (#2706)
* 增加了依据最高/最低指标列出 Trial 的[命令](https://github.com/microsoft/nni/blob/v1.8/docs/zh_CN/Tutorial/Nnictl.md#nnictl-trial) (#2747)
* 提升了 [nnicli](https://github.com/microsoft/nni/blob/v1.8/docs/zh_CN/nnicli_ref.md) 的用户体验，并附[示例](https://github.com/microsoft/nni/blob/v1.8/examples/notebooks/retrieve_nni_info_with_python.ipynb) (#2713)

### 神经网络架构搜索

* [搜索空间集合: ENAS 和 DARTS](https://github.com/microsoft/nni/blob/v1.8/docs/zh_CN/NAS/SearchSpaceZoo.md) (#2589)
* 用于在 NAS 基准测试中查询中间结果的 API (#2728)

### 模型压缩

* 支持 TorchModuleGraph 的 List/Tuple Construct/Unpack 操作 (#2609)
* 模型加速改进: 支持 DenseNet 和 InceptionV3 (#2719)
* 支持多个连续 tuple 的 unpack 操作 (#2768)
* [比较支持的 Pruner 的表现的文档](https://github.com/microsoft/nni/blob/v1.8/docs/zh_CN/CommunitySharings/ModelCompressionComparison.md) (#2742)
* 新的 Pruners: [Sensitivity pruner](https://github.com/microsoft/nni/blob/v1.8/docs/zh_CN/Compressor/Pruner.md#sensitivity-pruner) (#2684) and [AMC pruner](https://github.com/microsoft/nni/blob/v1.8/docs/zh_CN/Compressor/Pruner.md) (#2573) (#2786)
* 支持 TensorFlow v2 的模型压缩 (#2755)

### Backward incompatible changes

* Update the default experiment folder from `$HOME/nni/experiments` to `$HOME/nni-experiments`. If you want to view the experiments created by previous NNI releases, you can move the experiments folders from `$HOME/nni/experiments` to `$HOME/nni-experiments` manually. (#2686) (#2753)
* 不再支持 Python 3.5 和 scikit-learn 0.20 (#2778) (#2777) (2783) (#2787) (#2788) (#2790)

### 其它

* 更新 Docker 镜像中的 Tensorflow 版本 (#2732) (#2735) (#2720)

## 示例

* 在 Assessor 示例中移除 gpuNum (#2641)

## 文档

* 改进自定义 Tuner 的文档 (#2628)
* 修复几处文档中的输入错误和语法错误 (#2637 #2638, 感谢 @tomzx)
* 改进 AzureML 训练平台的文档 (#2631)
* 改进中文翻译的 CI 流程 (#2654)
* 改进 OpenPAI 训练平台的文档 (#2685)
* 改进社区分享的文档 (#2640)
* 增加对 Colab 进行支持的教程 (#2700)
* 改进模型压缩的文档结构 (#2676)

## 修复的 Bug

* 修复训练平台的目录生成错误 (#2673)
* 修复 Remote 训练平台使用 chmod 时的 Bug (#2689)
* 通过内联导入 `_graph_utils` 修复依赖问题 (#2675)
* 修复了 `SimulatedAnnealingPruner` 中的掩码问题 (#2736)
* 修复了中间结果的图的缩放问题 (#2738)
* 修复了在查询 NAS 基准测试时字典没有经过排序的问题 (#2728)
* Fix import issue for gradient selector dataloader iterator (#2690)
* Fix support of adding tens of machines in remote training service (#2725)
* Fix several styling issues in WebUI (#2762 #2737)
* Fix support of unusual types in metrics including NaN and Infinity (#2782)
* Fix nnictl experiment delete (#2791)

# Release 1.7 - 7/8/2020

## Major Features

### Training Service

* 支持 AML (Azure Machine Learning) 作为训练平台。
* OpenPAI 任务可被重用。 When a trial is completed, the OpenPAI job won't stop, and wait next trial. [refer to reuse flag in OpenPAI config](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/TrainingService/PaiMode.md#openpai-configurations).
* [Support ignoring files and folders in code directory with .nniignore when uploading code directory to training service](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/TrainingService/Overview.md#how-to-use-training-service).

### 神经网络架构搜索（NAS）

* [Provide NAS Open Benchmarks (NasBench101, NasBench201, NDS) with friendly APIs](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/NAS/Benchmarks.md).

* [Support Classic NAS (i.e., non-weight-sharing mode) on TensorFlow 2.X](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/NAS/ClassicNas.md).

### Model Compression

* Improve Model Speedup: track more dependencies among layers and automatically resolve mask conflict, support the speedup of pruned resnet.
* Added new pruners, including three auto model pruning algorithms: [NetAdapt Pruner](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Pruner.md#netadapt-pruner), [SimulatedAnnealing Pruner](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Pruner.md#simulatedannealing-pruner), [AutoCompress Pruner](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Pruner.md#autocompress-pruner), and [ADMM Pruner](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Pruner.md#admm-pruner).
* Added [model sensitivity analysis tool](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/CompressionUtils.md) to help users find the sensitivity of each layer to the pruning.
* [Easy flops calculation for model compression and NAS](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/CompressionUtils.md#model-flops-parameters-counter).

* Update lottery ticket pruner to export winning ticket.

### Examples

* Automatically optimize tensor operators on NNI with a new [customized tuner OpEvo](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/TrialExample/OpEvoExamples.md).

### Built-in tuners/assessors/advisors

* [Allow customized tuners/assessor/advisors to be installed as built-in algorithms](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Tutorial/InstallCustomizedAlgos.md).

### WebUI

* Support visualizing nested search space more friendly.
* Show trial's dict keys in hyper-parameter graph.
* Enhancements to trial duration display.

### Others

* Provide utility function to merge parameters received from NNI
* Support setting paiStorageConfigName in pai mode

## Documentation

* Improve [documentation for model compression](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/Compressor/Overview.md)
* Improve [documentation](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/NAS/Benchmarks.md) and [examples](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/NAS/BenchmarksExample.ipynb) for NAS benchmarks.
* Improve [documentation for AzureML training service](https://github.com/microsoft/nni/blob/v1.7/docs/en_US/TrainingService/AMLMode.md)
* Homepage migration to readthedoc.

## Bug Fixes

* Fix bug for model graph with shared nn.Module
* Fix nodejs OOM when `make build`
* Fix NASUI bugs
* Fix duration and intermediate results pictures update issue.
* Fix minor WebUI table style issues.

## Release 1.6 - 5/26/2020

### 主要功能

#### 新功能和改进

* Improve IPC limitation to 100W
* improve code storage upload logic among trials in non-local platform
* support `__version__` for SDK version
* support windows dev intall

#### Web 界面

* Show trial error message
* finalize homepage layout
* Refactor overview's best trials module
* Remove multiphase from webui
* add tooltip for trial concurrency in the overview page
* Show top trials for hyper-parameter graph

#### 超参优化更新

* Improve PBT on failure handling and support experiment resume for PBT

#### NAS 更新

* NAS support for TensorFlow 2.0 (preview) [TF2.0 NAS examples](https://github.com/microsoft/nni/tree/v1.9/examples/nas/naive-tf)
* Use OrderedDict for LayerChoice
* Prettify the format of export
* Replace layer choice with selected module after applied fixed architecture

#### 模型压缩改进

* Model compression PyTorch 1.4 support

#### 训练平台改进

* update pai yaml merge logic
* support windows as remote machine in remote mode [Remote Mode](https://github.com/microsoft/nni/blob/v1.9/docs/en_US/TrainingService/RemoteMachineMode.md#windows)

### Bug Fix

* fix dev install
* SPOS example crash when the checkpoints do not have state_dict
* Fix table sort issue when experiment had failed trial
* Support multi python env (conda, pyenv etc)

## Release 1.5 - 4/13/2020

### New Features and Documentation

#### 超参优化

* New tuner: [Population Based Training (PBT)](https://github.com/microsoft/nni/blob/v1.9/docs/en_US/Tuner/PBTTuner.md)
* Trials can now report infinity and NaN as result

#### 神经网络架构搜索

* New NAS algorithm: [TextNAS](https://github.com/microsoft/nni/blob/v1.9/docs/en_US/NAS/TextNAS.md)
* ENAS and DARTS now support [visualization](https://github.com/microsoft/nni/blob/v1.9/docs/en_US/NAS/Visualization.md) through web UI.

#### 模型压缩

* New Pruner: [GradientRankFilterPruner](https://github.com/microsoft/nni/blob/v1.9/docs/en_US/Compressor/Pruner.md#gradientrankfilterpruner)
* Compressors will validate configuration by default
* Refactor: Adding optimizer as an input argument of pruner, for easy support of DataParallel and more efficient iterative pruning. This is a broken change for the usage of iterative pruning algorithms.
* Model compression examples are refactored and improved
* Added documentation for [implementing compressing algorithm](https://github.com/microsoft/nni/blob/v1.9/docs/en_US/Compressor/Framework.md)

#### 训练平台

* Kubeflow now supports pytorchjob crd v1 (thanks external contributor @jiapinai)
* Experimental [DLTS](https://github.com/microsoft/nni/blob/v1.9/docs/en_US/TrainingService/DLTSMode.md) support

#### 文档的整体改进

* Documentation is significantly improved on grammar, spelling, and wording (thanks external contributor @AHartNtkn)

### Fixed Bugs

* ENAS cannot have more than one LSTM layers (thanks external contributor @marsggbo)
* NNI manager's timers will never unsubscribe (thanks external contributor @guilhermehn)
* NNI manager may exhaust head memory (thanks external contributor @Sundrops)
* Batch tuner does not support customized trials (#2075)
* Experiment cannot be killed if it failed on start (#2080)
* Non-number type metrics break web UI (#2278)
* A bug in lottery ticket pruner
* Other minor glitches

## Release 1.4 - 2/19/2020

### Major Features

#### 神经网络架构搜索

* Support [C-DARTS](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/NAS/CDARTS.md) algorithm and add [the example](https://github.com/microsoft/nni/tree/v1.4/examples/nas/cdarts) using it
* Support a preliminary version of [ProxylessNAS](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/NAS/Proxylessnas.md) and the corresponding [example](https://github.com/microsoft/nni/tree/v1.4/examples/nas/proxylessnas)
* Add unit tests for the NAS framework

#### 模型压缩

* Support DataParallel for compressing models, and provide [an example](https://github.com/microsoft/nni/blob/v1.4/examples/model_compress/multi_gpu.py) of using DataParallel
* Support [model speedup](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/Compressor/ModelSpeedup.md) for compressed models, in Alpha version

#### 训练平台

* Support complete PAI configurations by allowing users to specify PAI config file path
* Add example config yaml files for the new PAI mode (i.e., paiK8S)
* Support deleting experiments using sshkey in remote mode (thanks external contributor @tyusr)

#### Web 界面

* WebUI refactor: adopt fabric framework

#### 其它

* Support running [NNI experiment at foreground](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/Tutorial/Nnictl#manage-an-experiment), i.e., `--foreground` argument in `nnictl create/resume/view`
* Support canceling the trials in UNKNOWN state
* Support large search space whose size could be up to 50mb (thanks external contributor @Sundrops)

### Documentation

* Improve [the index structure](https://nni.readthedocs.io/en/latest/) of NNI readthedocs
* Improve [documentation for NAS](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/NAS/NasGuide.md)
* Improve documentation for [the new PAI mode](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/TrainingService/PaiMode.md)
* Add QuickStart guidance for [NAS](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/NAS/QuickStart.md) and [model compression](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/Compressor/QuickStart.md)
* Improve documentation for [the supported EfficientNet](https://github.com/microsoft/nni/blob/v1.4/docs/en_US/TrialExample/EfficientNet.md)

### Bug Fixes

* Correctly support NaN in metric data, JSON compliant
* Fix the out-of-range bug of `randint` type in search space
* Fix the bug of wrong tensor device when exporting onnx model in model compression
* Fix incorrect handling of nnimanagerIP in the new PAI mode (i.e., paiK8S)

## Release 1.3 - 12/30/2019

### Major Features

#### 支持神经网络架构搜索算法

* [Single Path One Shot](https://github.com/microsoft/nni/tree/v1.3/examples/nas/spos/) algorithm and the example using it

#### 模型压缩算法支持

* [Knowledge Distillation](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/TrialExample/KDExample.md) algorithm and the example using itExample
* Pruners 
    * [L2Filter Pruner](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Pruner.md#3-l2filter-pruner)
    * [ActivationAPoZRankFilterPruner](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Pruner.md#1-activationapozrankfilterpruner)
    * [ActivationMeanRankFilterPruner](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Pruner.md#2-activationmeanrankfilterpruner)
* [BNN Quantizer](https://github.com/microsoft/nni/blob/v1.3/docs/en_US/Compressor/Quantizer.md#bnn-quantizer)

#### 训练平台

* NFS Support for PAI
    
    Instead of using HDFS as default storage, since OpenPAI v0.11, OpenPAI can have NFS or AzureBlob or other storage as default storage. In this release, NNI extended the support for this recent change made by OpenPAI, and could integrate with OpenPAI v0.11 or later version with various default storage.

* Kubeflow update adoption
    
    Adopted the Kubeflow 0.7's new supports for tf-operator.

### Engineering (code and build automation)

* Enforced [ESLint](https://eslint.org/) on static code analysis.

### Small changes & Bug Fixes

* correctly recognize builtin tuner and customized tuner
* logging in dispatcher base
* fix the bug where tuner/assessor's failure sometimes kills the experiment.
* Fix local system as remote machine [issue](https://github.com/microsoft/nni/issues/1852)
* de-duplicate trial configuration in smac tuner [ticket](https://github.com/microsoft/nni/issues/1364)

## Release 1.2 - 12/02/2019

### 主要功能

* [Feature Engineering](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/FeatureEngineering/Overview.md) 
  - New feature engineering interface
  - Feature selection algorithms: [Gradient feature selector](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/FeatureEngineering/GradientFeatureSelector.md) & [GBDT selector](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/FeatureEngineering/GBDTSelector.md)
  - [Examples for feature engineering](https://github.com/microsoft/nni/tree/v1.2/examples/feature_engineering)
- Neural Architecture Search (NAS) on NNI 
  - [New NAS interface](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/NasInterface.md)
  - NAS algorithms: [ENAS](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/Overview.md#enas), [DARTS](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/Overview.md#darts), [P-DARTS](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/NAS/Overview.md#p-darts) (in PyTorch)
  - NAS in classic mode (each trial runs independently)
- Model compression 
  - [New model pruning algorithms](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/Compressor/Overview.md): lottery ticket pruning approach, L1Filter pruner, Slim pruner, FPGM pruner
  - [New model quantization algorithms](https://github.com/microsoft/nni/blob/v1.2/docs/en_US/Compressor/Overview.md): QAT quantizer, DoReFa quantizer
  - Support the API for exporting compressed model.
- Training Service 
  - Support OpenPAI token authentication
- Examples: 
  - [An example to automatically tune rocksdb configuration with NNI](https://github.com/microsoft/nni/tree/v1.2/examples/trials/systems/rocksdb-fillrandom).
  - [A new MNIST trial example supports tensorflow 2.0](https://github.com/microsoft/nni/tree/v1.2/examples/trials/mnist-tfv2).
- Engineering Improvements 
  - For remote training service, trial jobs require no GPU are now scheduled with round-robin policy instead of random.
  - Pylint rules added to check pull requests, new pull requests need to comply with these [pylint rules](https://github.com/microsoft/nni/blob/v1.2/pylintrc).
- Web Portal & User Experience 
  - Support user to add customized trial.
  - User can zoom out/in in detail graphs, except Hyper-parameter.
- Documentation 
  - Improved NNI API documentation with more API docstring.

### 修复的 Bug

- Fix the table sort issue when failed trials haven't metrics. -Issue #1773
- Maintain selected status(Maximal/Minimal) when the page switched. -PR#1710
- Make hyper-parameters graph's default metric yAxis more accurate. -PR#1736
- Fix GPU script permission issue. -Issue #1665

## Release 1.1 - 10/23/2019

### 主要功能

* New tuner: [PPO Tuner](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tuner/PPOTuner.md)
* [View stopped experiments](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tutorial/Nnictl.md#view)
* Tuners can now use dedicated GPU resource (see `gpuIndices` in [tutorial](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Tutorial/ExperimentConfig.md) for details)
* Web UI improvements 
  - Trials detail page can now list hyperparameters of each trial, as well as their start and end time (via "add column")
  - Viewing huge experiment is now less laggy
- More examples 
  - [EfficientNet PyTorch example](https://github.com/ultmaster/EfficientNet-PyTorch)
  - [Cifar10 NAS example](https://github.com/microsoft/nni/blob/v1.1/examples/trials/nas_cifar10/README.md)
- [Model compression toolkit - Alpha release](https://github.com/microsoft/nni/blob/v1.1/docs/en_US/Compressor/Overview.md): We are glad to announce the alpha release for model compression toolkit on top of NNI, it's still in the experiment phase which might evolve based on usage feedback. We'd like to invite you to use, feedback and even contribute

### Fixed Bugs

* Multiphase job hangs when search space exhuasted (issue #1204)
* `nnictl` fails when log not available (issue #1548)

## Release 1.0 - 9/2/2019

### 主要功能

* Tuners and Assessors
    
    - Support Auto-Feature generator & selection -Issue#877 -PR #1387 
        + Provide auto feature interface
        + Tuner based on beam search
        + [Add Pakdd example](https://github.com/microsoft/nni/tree/v1.9/examples/trials/auto-feature-engineering)
    + Add a parallel algorithm to improve the performance of TPE with large concurrency. -PR #1052
    + Support multiphase for hyperband -PR #1257
+ Training Service
    
    - Support private docker registry -PR #755
        
        * Engineering Improvements
        * Python wrapper for rest api, support retrieve the values of the metrics in a programmatic way PR #1318
        * New python API : get_experiment_id(), get_trial_id() -PR #1353 -Issue #1331 & -Issue#1368
        * Optimized NAS Searchspace -PR #1393 
         + Unify NAS search space with _type -- "mutable_type"e
         + Update random search tuner
        + Set gpuNum as optional -Issue #1365
        + Remove outputDir and dataDir configuration in PAI mode -Issue #1342
        + When creating a trial in Kubeflow mode, codeDir will no longer be copied to logDir -Issue #1224
+ Web Portal & User Experience
    
    - Show the best metric curve during search progress in WebUI -Issue #1218
    - Show the current number of parameters list in multiphase experiment -Issue1210 -PR #1348
    - Add "Intermediate count" option in AddColumn. -Issue #1210
    - Support search parameters value in WebUI -Issue #1208
    - Enable automatic scaling of axes for metric value in default metric graph -Issue #1360
    - Add a detailed documentation link to the nnictl command in the command prompt -Issue #1260
    - UX improvement for showing Error log -Issue #1173
- Documentation
    
    - Update the docs structure -Issue #1231
    - (deprecated) Multi phase document improvement -Issue #1233 -PR #1242 
        + Add configuration example
    + [WebUI description improvement](Tutorial/WebUI.md) -PR #1419

### Bug fix

* (Bug fix)Fix the broken links in 0.9 release -Issue #1236
* (Bug fix)Script for auto-complete
* (Bug fix)Fix pipeline issue that it only check exit code of last command in a script. -PR #1417
* (Bug fix)quniform fors tuners -Issue #1377
* (Bug fix)'quniform' has different meaning beween GridSearch and other tuner. -Issue #1335
* (Bug fix)"nnictl experiment list" give the status of a "RUNNING" experiment as "INITIALIZED" -PR #1388
* (Bug fix)SMAC cannot be installed if nni is installed in dev mode -Issue #1376
* (Bug fix)The filter button of the intermediate result cannot be clicked -Issue #1263
* (Bug fix)API "/api/v1/nni/trial-jobs/xxx" doesn't show a trial's all parameters in multiphase experiment -Issue #1258
* (Bug fix)Succeeded trial doesn't have final result but webui show ×××(FINAL) -Issue #1207
* (Bug fix)IT for nnictl stop -Issue #1298
* (Bug fix)fix security warning
* (Bug fix)Hyper-parameter page broken -Issue #1332
* (Bug fix)Run flake8 tests to find Python syntax errors and undefined names -PR #1217

## Release 0.9 - 7/1/2019

### 主要功能

* General NAS programming interface 
    * Add `enas-mode` and `oneshot-mode` for NAS interface: [PR #1201](https://github.com/microsoft/nni/pull/1201#issue-291094510)
* [Gaussian Process Tuner with Matern kernel](Tuner/GPTuner.md)

* (deprecated) Multiphase experiment supports
    
    * Added new training service support for multiphase experiment: PAI mode supports multiphase experiment since v0.9.
    * Added multiphase capability for the following builtin tuners: 
        * TPE, Random Search, Anneal, Naïve Evolution, SMAC, Network Morphism, Metis Tuner.
* Web Portal
    
    * Enable trial comparation in Web Portal. For details, refer to [View trials status](Tutorial/WebUI.md)
    * Allow users to adjust rendering interval of Web Portal. For details, refer to [View Summary Page](Tutorial/WebUI.md)
    * show intermediate results more friendly. For details, refer to [View trials status](Tutorial/WebUI.md)
* [Commandline Interface](Tutorial/Nnictl.md) 
    * `nnictl experiment delete`: delete one or all experiments, it includes log, result, environment information and cache. It uses to delete useless experiment result, or save disk space.
    * `nnictl platform clean`: It uses to clean up disk on a target platform. The provided YAML file includes the information of target platform, and it follows the same schema as the NNI configuration file.

### Bug fix and other changes

* Tuner Installation Improvements: add [sklearn](https://scikit-learn.org/stable/) to nni dependencies.
* (Bug Fix) Failed to connect to PAI http code - [Issue #1076](https://github.com/microsoft/nni/issues/1076)
* (Bug Fix) Validate file name for PAI platform - [Issue #1164](https://github.com/microsoft/nni/issues/1164)
* (Bug Fix) Update GMM evaluation in Metis Tuner
* (Bug Fix) Negative time number rendering in Web Portal - [Issue #1182](https://github.com/microsoft/nni/issues/1182), [Issue #1185](https://github.com/microsoft/nni/issues/1185)
* (Bug Fix) Hyper-parameter not shown correctly in WebUI when there is only one hyper parameter - [Issue #1192](https://github.com/microsoft/nni/issues/1192)

## Release 0.8 - 6/4/2019

### 主要功能

* Support NNI on Windows for OpenPAI/Remote mode 
  * NNI running on windows for remote mode
  * NNI running on windows for OpenPAI mode
* Advanced features for using GPU 
  * Run multiple trial jobs on the same GPU for local and remote mode
  * Run trial jobs on the GPU running non-NNI jobs
* Kubeflow v1beta2 operator 
  * Support Kubeflow TFJob/PyTorchJob v1beta2
* [General NAS programming interface](https://github.com/microsoft/nni/blob/v0.8/docs/en_US/GeneralNasInterfaces.md) 
  * Provide NAS programming interface for users to easily express their neural architecture search space through NNI annotation
  * Provide a new command `nnictl trial codegen` for debugging the NAS code
  * Tutorial of NAS programming interface, example of NAS on MNIST, customized random tuner for NAS
* Support resume tuner/advisor's state for experiment resume
* For experiment resume, tuner/advisor will be resumed by replaying finished trial data
* Web Portal 
  * Improve the design of copying trial's parameters
  * Support 'randint' type in hyper-parameter graph
  * Use should ComponentUpdate to avoid unnecessary render

### Bug fix and other changes

* Bug fix that `nnictl update` has inconsistent command styles
* Support import data for SMAC tuner
* Bug fix that experiment state transition from ERROR back to RUNNING
* Fix bug of table entries
* Nested search space refinement
* Refine 'randint' type and support lower bound
* [Comparison of different hyper-parameter tuning algorithm](CommunitySharings/HpoComparison.md)
* [Comparison of NAS algorithm](CommunitySharings/NasComparison.md)
* [NNI practice on Recommenders](CommunitySharings/RecommendersSvd.md)

## Release 0.7 - 4/29/2018

### Major Features

* [Support NNI on Windows](Tutorial/InstallationWin.md) 
  * NNI running on windows for local mode
* [New advisor: BOHB](Tuner/BohbAdvisor.md) 
  * Support a new advisor BOHB, which is a robust and efficient hyperparameter tuning algorithm, combines the advantages of Bayesian optimization and Hyperband
* [Support import and export experiment data through nnictl](Tutorial/Nnictl.md) 
  * Generate analysis results report after the experiment execution
  * Support import data to tuner and advisor for tuning
* [Designated gpu devices for NNI trial jobs](Tutorial/ExperimentConfig.md#localConfig) 
  * Specify GPU devices for NNI trial jobs by gpuIndices configuration, if gpuIndices is set in experiment configuration file, only the specified GPU devices are used for NNI trial jobs.
* Web Portal enhancement 
  * Decimal format of metrics other than default on the Web UI
  * Hints in WebUI about Multi-phase
  * Enable copy/paste for hyperparameters as python dict
  * Enable early stopped trials data for tuners.
* NNICTL provide better error message 
  * nnictl provide more meaningful error message for YAML file format error

### Bug fix

* Unable to kill all python threads after nnictl stop in async dispatcher mode
* nnictl --version does not work with make dev-install
* All trail jobs status stays on 'waiting' for long time on OpenPAI platform

## Release 0.6 - 4/2/2019

### Major Features

* [Version checking](TrainingService/PaiMode.md) 
  * check whether the version is consistent between nniManager and trialKeeper
* [Report final metrics for early stop job](https://github.com/microsoft/nni/issues/776) 
  * If includeIntermediateResults is true, the last intermediate result of the trial that is early stopped by assessor is sent to tuner as final result. The default value of includeIntermediateResults is false.
* [Separate Tuner/Assessor](https://github.com/microsoft/nni/issues/841) 
  * Adds two pipes to separate message receiving channels for tuner and assessor.
* Make log collection feature configurable
* Add intermediate result graph for all trials

### Bug fix

* [Add shmMB config key for OpenPAI](https://github.com/microsoft/nni/issues/842)
* Fix the bug that doesn't show any result if metrics is dict
* Fix the number calculation issue for float types in hyperband
* Fix a bug in the search space conversion in SMAC tuner
* Fix the WebUI issue when parsing experiment.json with illegal format
* Fix cold start issue in Metis Tuner

## Release 0.5.2 - 3/4/2019

### Improvements

* Curve fitting assessor performance improvement.

### Documentation

* Chinese version document: https://nni.readthedocs.io/zh/latest/
* Debuggability/serviceability document: https://nni.readthedocs.io/en/latest/Tutorial/HowToDebug.html
* Tuner assessor reference: https://nni.readthedocs.io/en/latest/sdk_reference.html

### Bug Fixes and Other Changes

* Fix a race condition bug that does not store trial job cancel status correctly.
* Fix search space parsing error when using SMAC tuner.
* Fix cifar10 example broken pipe issue.
* Add unit test cases for nnimanager and local training service.
* Add integration test azure pipelines for remote machine, OpenPAI and kubeflow training services.
* Support Pylon in OpenPAI webhdfs client.

## Release 0.5.1 - 1/31/2018

### Improvements

* Making [log directory](https://github.com/microsoft/nni/blob/v0.5.1/docs/ExperimentConfig.md) configurable
* Support [different levels of logs](https://github.com/microsoft/nni/blob/v0.5.1/docs/ExperimentConfig.md), making it easier for debugging

### Documentation

* Reorganized documentation & New Homepage Released: https://nni.readthedocs.io/en/latest/

### Bug Fixes and Other Changes

* Fix the bug of installation in python virtualenv, and refactor the installation logic
* Fix the bug of HDFS access failure on OpenPAI mode after OpenPAI is upgraded.
* Fix the bug that sometimes in-place flushed stdout makes experiment crash

## 发布 0.5.0 - 01/14/2019

### 主要功能

#### 支持新的 Tuner 和 Assessor

* 支持新的 [Metis Tuner](Tuner/MetisTuner.md)。 对于**在线**超参调优的场景，Metis 算法已经被证明非常有效。
* 支持 [ENAS customized tuner](https://github.com/countif/enas_nni)。由 GitHub 社区用户所贡献。它是神经网络的搜索算法，能够通过强化学习来学习神经网络架构，比 NAS 的性能更好。
* 支持 [Curve fitting （曲线拟合）Assessor](Assessor/CurvefittingAssessor.md)，通过曲线拟合的策略来实现提前终止 Trial。
* [权重共享的](https://github.com/microsoft/nni/blob/v0.5/docs/AdvancedNAS.md)高级支持：为 NAS Tuner 提供权重共享，当前支持 NFS。

#### 改进训练平台

* [FrameworkController 训练平台](TrainingService/FrameworkControllerMode.md)：支持使用在 Kubernetes 上使用 FrameworkController 运行。 
  * FrameworkController 是 Kubernetes 上非常通用的控制器（Controller），能用来运行基于各种机器学习框架的分布式作业，如 TensorFlow，Pytorch， MXNet 等。
  * NNI 为作业定义了统一而简单的规范。
  * 如何使用 FrameworkController 的 MNIST 示例。

#### 改进用户体验

* 为 OpenPAI, Kubeflow 和 FrameworkController 模式提供更好的日志支持。 
  * 改进后的日志架构能将尝试的 stdout/stderr 通过 HTTP POST 方式发送给 NNI 管理器。 NNI 管理器将 Trial 的 stdout/stderr 消息存储在本地日志文件中。
  * 在 WEB 界面上显示 Trial 日志的链接。
* 支持将最终结果显示为键值对。

## 发布 0.4.1 - 12/14/2018

### 主要功能

#### 支持新的 Tuner

* 支持新的 [network morphism](Tuner/NetworkmorphismTuner.md) Tuner。

#### 改进训练平台

* 将 [Kubeflow 训练平台](TrainingService/KubeflowMode.md)的依赖从 kubectl CLI 迁移到 [Kubernetes API](https://kubernetes.io/docs/concepts/overview/kubernetes-api/) 客户端。
* Kubeflow 训练服务支持 [Pytorch-operator](https://github.com/kubeflow/pytorch-operator)。
* 改进将本地代码文件上传到 OpenPAI HDFS 的性能。
* 修复 OpenPAI 在 WEB 界面的 Bug：当 OpenPAI 认证过期后，Web 界面无法更新 Trial 作业的状态。

#### 改进 NNICTL

* Show version information both in nnictl and WebUI. You can run **nnictl -v** to show your current installed NNI version

#### 改进 WEB 界面

* Enable modify concurrency number during experiment
* Add feedback link to NNI github 'create issue' page
* Enable customize top 10 trials regarding to metric numbers (largest or smallest)
* Enable download logs for dispatcher & nnimanager
* Enable automatic scaling of axes for metric number
* Update annotation to support displaying real choice in searchspace

### New examples

* [FashionMnist](https://github.com/microsoft/nni/tree/v1.9/examples/trials/network_morphism), work together with network morphism tuner
* [Distributed MNIST example](https://github.com/microsoft/nni/tree/v1.9/examples/trials/mnist-distributed-pytorch) written in PyTorch

## Release 0.4 - 12/6/2018

### Major Features

* [Kubeflow Training service](TrainingService/KubeflowMode.md) 
  * Support tf-operator
  * [Distributed trial example](https://github.com/microsoft/nni/tree/v1.9/examples/trials/mnist-distributed/dist_mnist.py) on Kubeflow
* [Grid search tuner](Tuner/GridsearchTuner.md)
* [Hyperband tuner](Tuner/HyperbandAdvisor.md)
* Support launch NNI experiment on MAC
* WebUI 
  * UI support for hyperband tuner
  * Remove tensorboard button
  * Show experiment error message
  * Show line numbers in search space and trial profile
  * Support search a specific trial by trial number
  * Show trial's hdfsLogPath
  * Download experiment parameters

### Others

* Asynchronous dispatcher
* Docker file update, add pytorch library
* Refactor 'nnictl stop' process, send SIGTERM to nni manager process, rather than calling stop Rest API.
* OpenPAI training service bug fix 
  * Support NNI Manager IP configuration(nniManagerIp) in OpenPAI cluster config file, to fix the issue that user’s machine has no eth0 device
  * File number in codeDir is capped to 1000 now, to avoid user mistakenly fill root dir for codeDir
  * Don’t print useless ‘metrics is empty’ log in OpenPAI job’s stdout. Only print useful message once new metrics are recorded, to reduce confusion when user checks OpenPAI trial’s output for debugging purpose
  * Add timestamp at the beginning of each log entry in trial keeper.

## 发布 0.3.0 - 11/2/2018

### NNICTL 的新功能和更新

* 支持同时运行多个 Experiment。
    
    在 v0.3 以前，NNI 仅支持一次运行一个 Experiment。 此版本开始，用户可以同时运行多个 Experiment。 每个 Experiment 都需要一个唯一的端口，第一个 Experiment 会像以前版本一样使用默认端口。 需要为其它 Experiment 指定唯一端口：
    
    ```bash
    nnictl create --port 8081 --config <config file path>
    ```

* 支持更新最大 Trial 的数量。 使用 `nnictl update --help` 了解详情。 或参考 [NNICTL](Tutorial/Nnictl.md) 查看完整帮助。

### API 的新功能和更新

* <span style="color:red"><strong>不兼容的改动</strong></span>：nn.get_parameters() 改为 nni.get_next_parameter。 所有以前版本的示例将无法在 v0.3 上运行，需要重新克隆 NNI 代码库获取新示例。 如果在自己的代码中使用了 NNI，也需要相应的更新。

* 新 API **nni.get_sequence_id()**。 每个 Trial 任务都会被分配一个唯一的序列数字，可通过 nni.get_sequence_id() API 来获取。
    
    ```bash
    git clone -b v0.3 https://github.com/microsoft/nni.git
    ```

* **nni.report_final_result(result)** API 对结果参数支持更多的数据类型。
    
    可用类型：
    
  * int
  * float
  * 包含有 'default' 键值的 dict，'default' 的值必须为 int 或 float。 dict 可以包含任何其它键值对。

### 支持新的 Tuner

* **Batch Tuner（批处理调参器）** 会执行所有超参组合，可被用来批量提交 Trial 任务。

### 新示例

* 公开的 NNI Docker 映像：
    
    ```bash
    docker pull msranni/nni:latest
    ```

* 新的 Trial 示例：[NNI Sklearn Example](https://github.com/microsoft/nni/tree/v1.9/examples/trials/sklearn)

* 新的竞赛示例：[Kaggle Competition TGS Salt Example](https://github.com/microsoft/nni/tree/v1.9/examples/trials/kaggle-tgs-salt)

### 其它

* 界面重构，参考[网页文档](Tutorial/WebUI.md)，了解如何使用新界面。
* 持续集成：NNI 已切换到 Azure pipelines。

## 发布 0.2.0 - 9/29/2018

### 主要功能

* 支持 [OpenPAI](https://github.com/microsoft/pai) (又称 pai) 训练平台 (参考[这里](TrainingService/PaiMode.md)来了解如何在 OpenPAI 下提交 NNI 任务) 
  * 支持 pai 模式的训练服务。 NNI Trial 可发送至 OpenPAI 集群上运行
  * NNI Trial 输出 (包括日志和模型文件) 会被复制到 OpenPAI 的 HDFS 中。
* 支持 [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) Tuner (参考[这里](Tuner/SmacTuner.md)，了解如何使用 SMAC Tuner) 
  * [SMAC](https://www.cs.ubc.ca/~hutter/papers/10-TR-SMAC.pdf) 基于 Sequential Model-Based Optimization (SMBO). 它会利用使用过的突出的模型（高斯随机过程模型），并将随机森林引入到SMBO中，来处理分类参数。 NNI 的 SMAC 通过包装 [SMAC3](https://github.com/automl/SMAC3) 来支持。
* 支持将 NNI 安装在 [conda](https://conda.io/docs/index.html) 和 Python 虚拟环境中。
* 其它 
  * 更新 ga squad 样例与相关文档
  * 用户体验改善及 Bug 修复

## 发布 0.1.0 - 9/10/2018 (首个版本)

首次发布 Neural Network Intelligence (NNI)。

### 主要功能

* 安装和部署 
  * 支持 pip 和源代码安装
  * 支持本机（包括多 GPU 卡）训练和远程多机训练模式
* Tuner ，Assessor 和 Trial 
  * 支持的自动机器学习算法包括： hyperopt_tpe, hyperopt_annealing, hyperopt_random, 和 evolution_tuner。
  * 支持 Assessor（提前终止）算法包括：medianstop。
  * 提供 Python API 来自定义 Tuner 和 Assessor
  * 提供 Python API 来包装 Trial 代码，以便能在 NNI 中运行
* Experiment 
  * 提供命令行工具 'nnictl' 来管理 Experiment
  * 提供网页界面来查看并管理 Experiment
* 持续集成 
  * 使用 Ubuntu 的 [travis-ci](https://github.com/travis-ci) 来支持持续集成
* 其它 
  * 支持简单的 GPU 任务调度