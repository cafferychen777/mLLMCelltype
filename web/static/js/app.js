// Modern mLLMCelltype Web Application - Vue.js 3
// Optimized initialization for Core Web Vitals

let appInitialized = false;
let initializationAttempts = 0;
const MAX_INITIALIZATION_ATTEMPTS = 10;

function showAppLoadError(message) {
    if (document.getElementById('appLoadError')) return;
    const appElement = document.getElementById('app');
    if (!appElement?.parentNode) return;
    const errorElement = document.createElement('div');
    errorElement.id = 'appLoadError';
    errorElement.className = 'app-load-error';
    errorElement.setAttribute('role', 'alert');
    errorElement.textContent = message;
    appElement.parentNode.insertBefore(errorElement, appElement);
}

async function apiRequest(url, options = {}) {
    const { method = 'GET', json, body, headers = {} } = options;
    const requestHeaders = { ...headers };
    let requestBody = body;
    if (json !== undefined) {
        requestHeaders['Content-Type'] = 'application/json';
        requestBody = JSON.stringify(json);
    }

    const response = await fetch(url, {
        method,
        headers: requestHeaders,
        body: requestBody,
        credentials: 'same-origin'
    });
    const contentType = response.headers.get('Content-Type') || '';
    const data = contentType.includes('application/json')
        ? await response.json()
        : await response.text();
    if (!response.ok) {
        const error = new Error(data?.error || `Request failed with status ${response.status}`);
        error.response = { status: response.status, data };
        throw error;
    }
    return { data, status: response.status, headers: response.headers };
}

// Initialize app when all dependencies are loaded
window.initializeApp = function () {
    if (appInitialized) return;

    if (typeof Vue === 'undefined') {
        console.error('Application dependencies are not loaded.');
        initializationAttempts += 1;
        if (initializationAttempts < MAX_INITIALIZATION_ATTEMPTS) {
            setTimeout(window.initializeApp, 1000);
        } else {
            showAppLoadError('The application could not load its browser dependencies. Check your connection and refresh the page.');
        }
        return;
    }

    // Proceed with app initialization
    initializeVueApp();
};

// Separate Vue app initialization for better code organization
function initializeVueApp() {
    if (appInitialized) return;

    const { createApp, ref, computed, onMounted, onBeforeUnmount, watch } = Vue;

    // Main Vue.js 3 Application
    const App = {
        setup() {
            // Reactive state
            const currentLang = ref('en');
            const currentStep = ref(0);
            const globalLoading = ref(false);
            const loadingMessage = ref('');
            const toasts = ref([]);

            // Upload state
            const isDragover = ref(false);
            const isUploading = ref(false);
            const uploadError = ref('');
            const uploadedFile = ref(null);
            const dataPreview = ref([]);
            const dataColumns = ref([]);

            // Configuration state
            const species = ref('human');
            const tissue = ref('');
            // Default thresholds relaxed to reduce LLM API calls while maintaining accuracy
            const consensusThreshold = ref(0.6);  // Lowered from 0.7
            const entropyThreshold = ref(1.2);    // Raised from 1.0
            const maxDiscussionRounds = ref(3);
            const consensusModel = ref('');
            const showCustomInput = ref(false);
            const showCustomSpeciesInput = ref(false);

            // Predefined species types
            const predefinedSpeciesList = ref([
                { value: 'human', labelEn: 'Human', labelZh: '人类' },
                { value: 'mouse', labelEn: 'Mouse', labelZh: '小鼠' },
                { value: 'rat', labelEn: 'Rat', labelZh: '大鼠' },
                { value: 'zebrafish', labelEn: 'Zebrafish', labelZh: '斑马鱼' },
                { value: 'drosophila', labelEn: 'Drosophila', labelZh: '果蝇' },
                { value: 'c.elegans', labelEn: 'C. elegans', labelZh: '线虫' },
                { value: 'arabidopsis', labelEn: 'Arabidopsis', labelZh: '拟南芥' },
                { value: 'xenopus', labelEn: 'Xenopus', labelZh: '非洲爪蟾' },
                { value: 'chicken', labelEn: 'Chicken', labelZh: '鸡' },
                { value: 'macaque', labelEn: 'Macaque', labelZh: '猕猴' },
                { value: 'pig', labelEn: 'Pig', labelZh: '猪' },
                { value: 'rabbit', labelEn: 'Rabbit', labelZh: '兔子' },
                { value: 'yeast', labelEn: 'Yeast', labelZh: '酵母' },
                { value: 'planarian', labelEn: 'Planarian', labelZh: '涡虫' }
            ]);

            // Predefined tissue types
            const predefinedTissuesList = ref([
                { value: 'Blood', labelEn: 'Blood', labelZh: '血液' },
                { value: 'Brain', labelEn: 'Brain', labelZh: '大脑' },
                { value: 'Lung', labelEn: 'Lung', labelZh: '肺' },
                { value: 'Heart', labelEn: 'Heart', labelZh: '心脏' },
                { value: 'Liver', labelEn: 'Liver', labelZh: '肝脏' },
                { value: 'Kidney', labelEn: 'Kidney', labelZh: '肾脏' },
                { value: 'Muscle', labelEn: 'Muscle', labelZh: '肌肉' },
                { value: 'Skin', labelEn: 'Skin', labelZh: '皮肤' },
                { value: 'Bone', labelEn: 'Bone', labelZh: '骨骼' },
                { value: 'Adipose', labelEn: 'Adipose', labelZh: '脂肪组织' },
                { value: 'Pancreas', labelEn: 'Pancreas', labelZh: '胰腺' },
                { value: 'Stomach', labelEn: 'Stomach', labelZh: '胃' },
                { value: 'Intestine', labelEn: 'Intestine', labelZh: '肠道' },
                { value: 'Colon', labelEn: 'Colon', labelZh: '结肠' },
                { value: 'Breast', labelEn: 'Breast', labelZh: '乳腺' },
                { value: 'Prostate', labelEn: 'Prostate', labelZh: '前列腺' },
                { value: 'Ovary', labelEn: 'Ovary', labelZh: '卵巢' },
                { value: 'Testis', labelEn: 'Testis', labelZh: '睾丸' },
                { value: 'Thyroid', labelEn: 'Thyroid', labelZh: '甲状腺' },
                { value: 'Adrenal', labelEn: 'Adrenal', labelZh: '肾上腺' },
                { value: 'Spleen', labelEn: 'Spleen', labelZh: '脾脏' },
                { value: 'Lymph Node', labelEn: 'Lymph Node', labelZh: '淋巴结' },
                { value: 'Bone Marrow', labelEn: 'Bone Marrow', labelZh: '骨髓' },
                { value: 'Placenta', labelEn: 'Placenta', labelZh: '胎盘' },
                { value: 'Embryo', labelEn: 'Embryo', labelZh: '胚胎' },
                { value: 'Retina', labelEn: 'Retina', labelZh: '视网膜' },
                { value: 'Cornea', labelEn: 'Cornea', labelZh: '角膜' },
                { value: 'Spinal Cord', labelEn: 'Spinal Cord', labelZh: '脊髓' }
            ]);

            // Organism-tissue compatibility constraints
            // Define which tissues are NOT compatible with specific organisms
            const incompatibleCombinations = {
                // Yeast (single-celled fungus) - no tissues/organs
                'yeast': ['Blood', 'Brain', 'Lung', 'Heart', 'Liver', 'Kidney', 'Muscle', 'Skin',
                         'Bone', 'Adipose', 'Pancreas', 'Stomach', 'Intestine', 'Colon',
                         'Breast', 'Prostate', 'Ovary', 'Testis', 'Thyroid', 'Adrenal',
                         'Spleen', 'Lymph Node', 'Bone Marrow', 'Placenta', 'Embryo',
                         'Retina', 'Cornea', 'Spinal Cord'],

                // Arabidopsis (plant) - no animal tissues/organs
                'arabidopsis': ['Blood', 'Brain', 'Lung', 'Heart', 'Liver', 'Kidney', 'Muscle', 'Skin',
                               'Bone', 'Adipose', 'Pancreas', 'Stomach', 'Intestine', 'Colon',
                               'Breast', 'Prostate', 'Ovary', 'Testis', 'Thyroid', 'Adrenal',
                               'Spleen', 'Lymph Node', 'Bone Marrow', 'Placenta', 'Embryo',
                               'Retina', 'Cornea', 'Spinal Cord'],

                // C. elegans (simple nematode) - no complex organs, only basic tissues
                'c.elegans': ['Lung', 'Heart', 'Liver', 'Kidney', 'Bone', 'Adipose', 'Pancreas',
                             'Stomach', 'Colon', 'Breast', 'Prostate', 'Ovary', 'Testis',
                             'Thyroid', 'Adrenal', 'Spleen', 'Lymph Node', 'Bone Marrow',
                             'Placenta', 'Retina', 'Cornea', 'Spinal Cord'],

                // Planarian (simple flatworm) - similar to C. elegans
                'planarian': ['Lung', 'Heart', 'Liver', 'Kidney', 'Bone', 'Adipose', 'Pancreas',
                             'Stomach', 'Colon', 'Breast', 'Prostate', 'Ovary', 'Testis',
                             'Thyroid', 'Adrenal', 'Spleen', 'Lymph Node', 'Bone Marrow',
                             'Placenta', 'Retina', 'Cornea', 'Spinal Cord'],

                // Drosophila (fruit fly) - insect, lacks many vertebrate-specific organs
                'drosophila': ['Bone', 'Adipose', 'Breast', 'Prostate', 'Ovary', 'Testis',
                              'Thyroid', 'Adrenal', 'Spleen', 'Lymph Node', 'Bone Marrow',
                              'Placenta', 'Lung', 'Liver', 'Kidney', 'Pancreas', 'Stomach',
                              'Colon', 'Spinal Cord']
            };

            // Computed property: filtered tissue list based on selected species
            const filteredTissuesList = computed(() => {
                if (!species.value) {
                    return predefinedTissuesList.value;
                }

                const incompatible = incompatibleCombinations[species.value] || [];
                return predefinedTissuesList.value.filter(
                    tissue => !incompatible.includes(tissue.value)
                );
            });

            // Watch for species changes and clear incompatible tissue selection
            watch(species, (newSpecies) => {
                if (newSpecies && tissue.value) {
                    const incompatible = incompatibleCombinations[newSpecies] || [];
                    if (incompatible.includes(tissue.value)) {
                        const warningMsg = t.value('tissueIncompatible', { tissue: tissue.value, species: newSpecies });
                        tissue.value = '';
                        showToast(warningMsg, 'warning', 5000);
                    }
                }
            });

            // Processing state
            const taskId = ref(null);
            const taskStatus = ref('Preparing');
            const processingError = ref('');
            const progressPercent = ref(0);
            const processingTime = ref(0);
            const progressDetails = ref(null);
            const notificationSent = ref(false);
            const completedProcessing = ref(false);

            // Notification permission state
            const notificationPermission = ref(
                typeof window !== 'undefined' && 'Notification' in window
                    ? Notification.permission
                    : 'denied'
            );

            // Results state
            const results = ref({});

            // .env file import state
            const envImportMessage = ref(null);
            const isEnvDragover = ref(false);

            // Polling timer
            let pollTimer = null;
            let processingTimer = null;

            // Translations
            const translations = {
                en: {
                    // Welcome
                    heroEyebrow: 'Open-source scRNA-seq annotation',
                    welcomeTitle: 'Cell type annotation with evidence, not guesswork',
                    welcomeSubtitle: 'Compare leading language models on your marker genes and resolve agreement in one reproducible workflow.',
                    heroDescription: 'Review model-level outputs, consensus scores, and discussion logs before exporting your annotations.',
                    featureMultiModel: 'Independent model opinions',
                    featureMultiModelDesc: 'Run current models from multiple providers against the same marker-gene context.',
                    featureConsensus: 'Transparent consensus',
                    featureConsensusDesc: 'See agreement and entropy scores instead of receiving a single opaque prediction.',
                    featureSecurity: 'Session-scoped workflow',
                    featureSecurityDesc: 'API credentials are redacted from persisted task data and used only to run your analysis.',
                    getStarted: 'Start annotation',

                    // Related Resources
                    resourcesTitle: 'Documentation and source',
                    resourcesDescription: 'Learn the workflow, inspect the implementation, or report an issue.',
                    faqTitle: 'Frequently Asked Questions',
                    faqDescription: 'Get answers to common questions about cell type annotation and AI-powered analysis',
                    faqBadge: 'FAQ',

                    // Steps
                    stepUpload: 'Upload',
                    stepConfigure: 'Configure',
                    stepProcess: 'Process',
                    stepResults: 'Results',
                    progressLabel: 'Annotation progress',

                    // Upload
                    uploadTitle: 'Upload Marker Genes for Cell Type Annotation',
                    uploadDescription: 'Start cell type annotation by uploading marker genes from your scRNA-seq analysis. We support CSV, TSV, and Excel formats.',
                    dragDropTitle: 'Drop your file here',
                    dragDropText: 'or click anywhere in this area to select a file (CSV, TSV, Excel, max 16MB)',
                    selectFile: 'Choose File',
                    uploadingTitle: 'Processing Upload',
                    uploadingText: 'Validating and preparing your data...',
                    uploadSuccessTitle: 'Upload Successful',
                    fileName: 'File Name',
                    fileSize: 'File Size',
                    removeFile: 'Remove File',
                    supportedFormats: 'Supported Formats',
                    needSampleData: 'Need Sample Data?',
                    downloadSample: 'Download Sample',
                    dataPreview: 'Data Preview',
                    uploadError: 'Upload Error',

                    // Configuration
                    enableNotifications: 'Enable notifications to get alerted when your analysis completes',
                    enableNotificationsBtn: 'Enable Notifications',
                    notificationsEnabled: 'Notifications enabled! You\'ll be notified when annotation completes.',
                    notificationsDenied: 'Notifications were denied. You can enable them in your browser settings.',
                    configTitle: 'Configure Cell Type Annotation',
                    configDescription: 'Set up your scRNA-seq analysis parameters and select AI models for automated cell type annotation.',
                    basicSettings: 'Basic Settings',
                    species: 'Species',
                    clearSpecies: 'Clear species',
                    speciesPlaceholder: 'Enter custom species or select from above',
                    speciesHelp: 'Select a common species or enter a custom one. Providing species context improves annotation accuracy.',
                    tissue: 'Tissue Type',
                    clearTissue: 'Clear tissue',
                    tissuePlaceholder: 'Enter custom tissue type or select from above',
                    tissueHelp: 'Select a common tissue type or enter a custom one. Providing tissue context improves annotation accuracy.',
                    tissueIncompatible: 'The tissue "{tissue}" is not compatible with {species}. Tissue selection has been cleared.',
                    modelSelection: 'AI Model Selection',
                    selectModels: 'Select Models',
                    selectedModelsCount: 'Selected {count} model(s)',
                    customModelPlaceholder: 'Enter another provider model ID',
                    addModel: 'Add Model',
                    customModelRequired: 'Enter a model ID first',
                    customModelTooLong: 'Model IDs must be 300 characters or fewer',
                    apiKey: 'API Key',
                    apiKeyPlaceholder: 'Enter your API key',
                    apiKeyHelp: 'API keys are only used for this session',
                    getApiKey: 'Get API key',
                    modelWarning: 'Please select at least one AI model',
                    modelSelectionWarning: 'Please select at least one model for each enabled provider',
                    advancedOptions: 'Advanced Options',
                    consensusThreshold: 'Consensus Threshold',
                    consensusThresholdHelp: 'Minimum agreement proportion between models (0.1-1.0)',
                    consensusTipTitle: 'Tip: Improve consensus algorithm effectiveness',
                    consensusTipMessage: 'The consensus algorithm works best with at least 3 models. Consider selecting more models from different providers for more robust cell type annotations.',
                    entropyThreshold: 'Entropy Threshold',
                    entropyThresholdHelp: 'Maximum entropy value for accepting consensus (0.1-2.0)',
                    maxRounds: 'Max Discussion Rounds',
                    maxRoundsHelp: 'Maximum rounds of inter-model discussion (1-5)',
                    consensusModel: 'Consensus Check Model',
                    consensusModelHelp: 'Select a specific model for consensus checking and discussion (optional)',
                    consensusModelDefault: 'Auto-select (recommended)',

                    // Processing
                    processingTitle: 'Performing Cell Type Annotation',
                    processingDescription: 'AI models are analyzing your marker genes and building consensus on cell type annotations.',
                    selectedModels: 'Selected Models',

                    // Results
                    resultsTitle: 'Cell Type Annotation Results',
                    resultsDescription: 'Your scRNA annotation is complete. Download annotated cell types with confidence scores.',
                    clustersAnnotated: 'Clusters Annotated',
                    modelsUsed: 'Models Used',
                    processingTime: 'Processing Time',
                    annotationResults: 'Annotation Results',
                    cluster: 'Cluster',
                    cellType: 'Cell Type',
                    consensus: 'Consensus',
                    entropy: 'Entropy',
                    confidence: 'Confidence',
                    confidenceHigh: 'High',
                    confidenceMedium: 'Medium',
                    confidenceLow: 'Low',
                    analysisCompleted: 'Analysis completed successfully!',
                    notificationTitle: 'Cell Type Annotation Complete!',
                    notificationBody: 'Your analysis of {filename} is ready. {count} clusters annotated.',
                    newAnnotation: 'New Analysis',
                    rerunWithParams: 'Adjust Parameters & Rerun',
                    readyToRerun: 'Ready to adjust parameters and rerun',
                    readyForNewUpload: 'Ready to upload new data',
                    downloadResults: 'Download Results',
                    downloadLogs: 'Download Logs',
                    downloadDiscussion: 'Download Discussion',

                    // Common
                    dismiss: 'Dismiss',
                    back: 'Back',
                    continue: 'Continue',
                    startAnnotation: 'Start Analysis',

                    // Footer
                    learningResources: 'Learning Resources',
                    troubleshootingGuide: 'Troubleshooting Guide',
                    researchCommunity: 'Research & Community',
                    support: 'Support',
                    githubRepository: 'GitHub Repository',
                    githubRepositoryTitle: 'GitHub Repository',
                    githubRepositoryDescription: 'View source code, contribute, and report issues',
                    openSourceBadge: 'Open Source',
                    researchPaper: 'Research Paper',
                    sampleData: 'Sample Data',
                    reportIssues: 'Report Issues',
                    discussions: 'Discussions',
                    lastUpdated: 'Last updated',
                    loading: 'Loading...',
                    switchToChinese: 'Switch to Chinese',
                    switchToEnglish: 'Switch to English',
                    languageSwitched: 'Language switched to English',
                    viewOnGithub: 'View on GitHub',
                    readThePaper: 'Read the paper',
                    modelsText: 'models',
                    footerDescription: 'Open-source scRNA-seq cell type annotation tool using multiple AI models with consensus-based cell type identification.',
                    footerCopyright: 'Open source bioinformatics tool for the research community.',
                    apiDocs: 'API Documentation',

                    // Progress details
                    clusters: 'Clusters',
                    stage: 'Stage',
                    phase: 'Phase',
                    modelAnnotation: 'Model Annotation',
                    consensusChecking: 'Consensus Checking',
                    controversyResolution: 'Controversy Resolution',
                    phaseStarting: 'Starting',
                    processingError: 'Processing Error',

                    // Status
                    statusFileReady: 'File Ready',
                    statusQueued: 'Queued',
                    statusProcessing: 'Processing',
                    statusCompleted: 'Completed',
                    statusFailed: 'Failed',
                    statusTimeout: 'Timed Out',
                    statusCancelled: 'Cancelled',

                    // API Testing
                    testApiKey: 'Test Connection',
                    apiKeyValid: 'API key is valid',
                    apiKeyTestSuccess: 'API key test successful',
                    apiKeyTestFailed: 'API key test failed',
                    apiKeyRequired: 'Please enter API key first',
                    testAllApiKeys: 'Test All API Keys',
                    noApiKeysToTest: 'No API keys to test',
                    testing: 'Testing...',
                    justNow: 'Just now',
                    minutesAgo: '{minutes} minutes ago',
                    hoursAgo: '{hours} hours ago',
                    batchTestHelp: 'Test all configured API keys at once to verify they work properly',
                    showApiKey: 'Show API key',
                    hideApiKey: 'Hide API key',

                    // .env Import
                    importApiKeys: 'Import API Keys',
                    importApiKeysDesc: 'Drag & drop or click to upload a .env file to import all your API keys',
                    uploadEnvFile: 'Upload .env File',
                    downloadSampleEnv: 'Sample',
                    envImportSuccess: 'Successfully imported {count} API keys',
                    envImportError: 'Error reading .env file',
                    envImportPartial: 'Imported {success} keys, {failed} failed',
                    dropEnvFileHere: 'Drop .env file here to import API keys',
                    envFileTypeError: 'Please upload a .env file',

                    // Error messages
                    errorInvalidApiKey: '{provider} API key is invalid. Please check your key.',
                    errorRateLimit: '{provider} rate limit exceeded. Please try again later.',
                    errorInsufficientPermissions: '{provider} API key has insufficient permissions.',
                    errorTimeout: '{provider} request timed out. Please check your connection.',
                    errorConnection: 'Cannot connect to {provider} servers. Please try again.',
                    errorQuotaExceeded: '{provider} quota exceeded. Please check your billing.',
                    errorGeneric: '{provider} error: {error}',

                    // File upload errors
                    errorUnsupportedFileType: 'Unsupported file type: {ext}. Please upload CSV, TSV or Excel file.',
                    errorFileTooLarge: 'File is too large, please upload a file smaller than 16MB.',
                    errorUploadFailed: 'Upload failed, please try again.',
                    uploadSuccess: 'File uploaded successfully!',
                    uploadFailed: 'Upload failed',

                    // Processing errors
                    errorProcessingFailed: 'Processing failed.',
                    errorProcessingTimeout: 'Processing timed out. Please try again.',
                    errorStartAnalysis: 'Failed to start annotation task.',
                    analysisStarted: 'Analysis started successfully!',
                    analysisFailed: 'Analysis failed',
                    analysisTimeout: 'Analysis timed out',
                    analysisCancelled: 'Task cancelled',
                    errorTaskCancelled: 'Task was cancelled',

                    // Results errors
                    errorInvalidResultsFormat: 'Invalid results format.',
                    errorGetResults: 'Failed to get results.',
                    getResultsFailed: 'Failed to get results',
                    errorResetTask: 'Failed to reset task',
                    downloadFailed: 'Download failed.',
                    persistenceWarning: 'Results may not survive a server restart. Please download them now.'
                },
                zh: {
                    // Welcome
                    heroEyebrow: '开源 scRNA-seq 注释工具',
                    welcomeTitle: '用证据达成细胞类型注释共识',
                    welcomeSubtitle: '让多个主流语言模型分析同一组标记基因，并在一个可复现流程中比较和汇总结果。',
                    heroDescription: '导出注释前，可查看单模型结果、共识得分、熵值与讨论记录。',
                    featureMultiModel: '独立模型意见',
                    featureMultiModelDesc: '让多个提供商的当前模型在相同标记基因上下文中独立判断。',
                    featureConsensus: '透明的共识过程',
                    featureConsensusDesc: '直接查看一致率与熵值，而不是只得到一个不透明的结论。',
                    featureSecurity: '会话级工作流',
                    featureSecurityDesc: 'API 凭据不会写入持久化任务数据，只用于运行本次分析。',
                    getStarted: '开始注释',

                    // Related Resources
                    resourcesTitle: '文档与源代码',
                    resourcesDescription: '了解工作流、检查实现细节，或反馈问题。',
                    faqTitle: '常见问题解答',
                    faqDescription: '获取关于细胞类型注释和AI驱动分析的常见问题答案',
                    faqBadge: '常见问题',

                    // Steps
                    stepUpload: '上传',
                    stepConfigure: '配置',
                    stepProcess: '处理',
                    stepResults: '结果',
                    progressLabel: '注释进度',

                    // Upload
                    uploadTitle: '上传标记基因数据',
                    uploadDescription: '上传您的scRNA-seq分析中识别出的标记基因进行细胞类型注释。我们支持CSV、TSV和Excel格式。',
                    dragDropTitle: '将文件拖放到此处',
                    dragDropText: '或点击此区域任意位置选择文件（CSV、TSV、Excel格式，最大16MB）',
                    selectFile: '选择文件',
                    uploadingTitle: '处理上传',
                    uploadingText: '正在验证和准备您的数据...',
                    uploadSuccessTitle: '上传成功',
                    fileName: '文件名',
                    fileSize: '文件大小',
                    removeFile: '移除文件',
                    supportedFormats: '支持的格式',
                    needSampleData: '需要示例数据？',
                    downloadSample: '下载示例',
                    dataPreview: '数据预览',
                    uploadError: '上传错误',

                    // Configuration
                    enableNotifications: '启用通知，在分析完成时获得提醒',
                    enableNotificationsBtn: '启用通知',
                    notificationsEnabled: '通知已启用！注释完成时您将收到通知。',
                    notificationsDenied: '通知被拒绝。您可以在浏览器设置中启用它们。',
                    configTitle: '配置分析',
                    configDescription: '设置分析参数并选择用于注释的AI模型。',
                    basicSettings: '基本设置',
                    species: '物种',
                    clearSpecies: '清除物种',
                    speciesPlaceholder: '输入自定义物种或从上方选择',
                    speciesHelp: '选择常见物种或输入自定义物种。提供物种上下文可提高注释准确性。',
                    tissue: '组织类型',
                    clearTissue: '清除组织',
                    tissuePlaceholder: '输入自定义组织类型或从上方选择',
                    tissueHelp: '选择常见组织类型或输入自定义类型。提供组织上下文可提高注释准确性。',
                    tissueIncompatible: '组织"{tissue}"与{species}不兼容。组织选择已被清除。',
                    modelSelection: 'AI模型选择',
                    selectModels: '选择模型',
                    selectedModelsCount: '已选择 {count} 个模型',
                    customModelPlaceholder: '输入其他提供商模型 ID',
                    addModel: '添加模型',
                    customModelRequired: '请先输入模型 ID',
                    customModelTooLong: '模型 ID 不能超过 300 个字符',
                    apiKey: 'API密钥',
                    apiKeyPlaceholder: '输入您的API密钥',
                    apiKeyHelp: 'API密钥仅用于此会话',
                    getApiKey: '获取API密钥',
                    modelWarning: '请至少选择一个AI模型',
                    modelSelectionWarning: '请为每个启用的提供商至少选择一个模型',
                    advancedOptions: '高级选项',
                    consensusThreshold: '共识阈值',
                    consensusThresholdHelp: '模型间最小一致性比例 (0.1-1.0)',
                    consensusTipTitle: '提示：提高共识算法效果',
                    consensusTipMessage: '共识算法在使用至少3个模型时效果最佳。建议从不同提供商选择更多模型，以获得更稳健的细胞类型注释结果。',
                    entropyThreshold: '熵阈值',
                    entropyThresholdHelp: '接受共识的最大熵值 (0.1-2.0)',
                    maxRounds: '最大讨论轮数',
                    maxRoundsHelp: '模型间讨论的最大轮数 (1-5)',
                    consensusModel: '共识检查模型',
                    consensusModelHelp: '选择用于共识检查和讨论的特定模型 (可选)',
                    consensusModelDefault: '自动选择 (推荐)',

                    // Processing
                    processingTitle: 'AI处理中',
                    processingDescription: '所选AI模型正在分析您的数据并就细胞类型注释达成共识。',
                    selectedModels: '选择的模型',

                    // Results
                    resultsTitle: '注释结果',
                    resultsDescription: '分析完成！这是您的细胞类型注释及置信度指标。',
                    clustersAnnotated: '已注释集群',
                    modelsUsed: '使用的模型',
                    processingTime: '处理时间',
                    annotationResults: '注释结果',
                    cluster: '集群',
                    cellType: '细胞类型',
                    consensus: '共识',
                    entropy: '熵',
                    confidence: '置信度',
                    confidenceHigh: '高',
                    confidenceMedium: '中',
                    confidenceLow: '低',
                    analysisCompleted: '分析已成功完成！',
                    notificationTitle: '细胞类型注释完成！',
                    notificationBody: '{filename} 的分析已完成，共注释 {count} 个集群。',
                    newAnnotation: '新分析',
                    rerunWithParams: '调整参数重新分析',
                    readyToRerun: '准备调整参数并重新运行',
                    readyForNewUpload: '准备上传新数据',
                    downloadResults: '下载结果',
                    downloadLogs: '下载日志',
                    downloadDiscussion: '下载讨论详情',

                    // Common
                    dismiss: '关闭',
                    back: '返回',
                    continue: '继续',
                    startAnnotation: '开始分析',

                    // Footer
                    learningResources: '学习资源',
                    troubleshootingGuide: '故障排除指南',
                    researchCommunity: '研究与社区',
                    support: '支持',
                    githubRepository: 'GitHub 仓库',
                    githubRepositoryTitle: 'GitHub 仓库',
                    githubRepositoryDescription: '查看源代码、贡献代码和报告问题',
                    openSourceBadge: '开源项目',
                    researchPaper: '研究论文',
                    sampleData: '示例数据',
                    reportIssues: '报告问题',
                    discussions: '讨论区',
                    lastUpdated: '最后更新',
                    loading: '加载中...',
                    switchToChinese: '切换到中文',
                    switchToEnglish: '切换到英文',
                    languageSwitched: '语言已切换为中文',
                    viewOnGithub: '在GitHub上查看',
                    readThePaper: '阅读论文',
                    modelsText: '个模型',
                    footerDescription: '使用多个大型语言模型进行细胞类型注释的开源工具，通过共识机制进行细胞类型识别。',
                    footerCopyright: '面向研究社区的开源生物信息学工具。',
                    apiDocs: 'API 文档',

                    // Progress details
                    clusters: '簇',
                    stage: '阶段',
                    phase: '阶段',
                    modelAnnotation: '模型注释',
                    consensusChecking: '共识检查',
                    controversyResolution: '争议解决',
                    phaseStarting: '开始',
                    processingError: '处理错误',

                    // Status
                    statusFileReady: '文件已准备',
                    statusQueued: '排队中',
                    statusProcessing: '处理中',
                    statusCompleted: '已完成',
                    statusFailed: '失败',
                    statusTimeout: '超时',
                    statusCancelled: '已取消',

                    // API Testing
                    testApiKey: '测试连接',
                    apiKeyValid: 'API密钥有效',
                    apiKeyTestSuccess: 'API密钥测试成功',
                    apiKeyTestFailed: 'API密钥测试失败',
                    apiKeyRequired: '请先输入API密钥',
                    testAllApiKeys: '测试所有API密钥',
                    noApiKeysToTest: '没有需要测试的API密钥',
                    testing: '测试中...',
                    justNow: '刚刚',
                    minutesAgo: '{minutes}分钟前',
                    hoursAgo: '{hours}小时前',
                    batchTestHelp: '一键测试所有已配置的API密钥，确保它们能正常工作',
                    showApiKey: '显示密钥',
                    hideApiKey: '隐藏密钥',

                    // .env import
                    importApiKeys: '导入 API 密钥',
                    importApiKeysDesc: '拖放或点击上传 .env 文件，快速导入所有 API 密钥',
                    uploadEnvFile: '上传 .env 文件',
                    downloadSampleEnv: '示例',
                    envImportSuccess: '成功导入 {count} 个 API 密钥',
                    envImportError: '读取 .env 文件出错',
                    envImportPartial: '导入 {success} 个密钥，{failed} 个失败',
                    dropEnvFileHere: '拖放 .env 文件到此处导入 API 密钥',
                    envFileTypeError: '请上传 .env 文件',

                    // Error messages
                    errorInvalidApiKey: '{provider} API密钥无效，请检查您的密钥。',
                    errorRateLimit: '{provider} 调用频率超限，请稍后再试。',
                    errorInsufficientPermissions: '{provider} API密钥权限不足。',
                    errorTimeout: '{provider} 请求超时，请检查网络连接。',
                    errorConnection: '无法连接到{provider}服务器，请重试。',
                    errorQuotaExceeded: '{provider} 配额已用完，请检查账单。',
                    errorGeneric: '{provider} 错误：{error}',

                    // Upload errors
                    errorUnsupportedFileType: '不支持的文件类型：{ext}。请上传 CSV、TSV 或 Excel 文件。',
                    errorFileTooLarge: '文件过大，请上传小于 16MB 的文件。',
                    errorUploadFailed: '上传失败，请重试。',
                    uploadSuccess: '文件上传成功！',
                    uploadFailed: '上传失败',

                    // Processing errors
                    errorProcessingFailed: '处理失败。',
                    errorProcessingTimeout: '处理超时，请重试。',
                    errorStartAnalysis: '启动注释任务失败。',
                    analysisStarted: '分析已成功启动！',
                    analysisFailed: '分析失败',
                    analysisTimeout: '分析超时',
                    analysisCancelled: '任务已取消',
                    errorTaskCancelled: '任务已取消',

                    // Result errors
                    errorInvalidResultsFormat: '无效的结果格式。',
                    errorGetResults: '获取结果失败。',
                    getResultsFailed: '获取结果失败',
                    errorResetTask: '重置任务失败',
                    downloadFailed: '下载失败。',
                    persistenceWarning: '结果可能无法在服务器重启后保留，请立即下载。'
                }
            };

            const createProviderState = (id, name) => ({
                id,
                name,
                selected: false,
                apiKey: '',
                selectedModels: [],
                customModel: '',
                showApiKey: false,
                testing: false,
                testResult: null,
                lastTested: null,
                models: []
            });
            const availableProviders = ref([]);
            // Computed properties
            const t = computed(() => {
                return (key, params = {}) => {
                    let text = translations[currentLang.value][key] || key;
                    // Replace parameters in the text
                    Object.keys(params).forEach(param => {
                        text = text.replace(new RegExp(`{${param}}`, 'g'), params[param]);
                    });
                    return text;
                };
            });

            const hasSelectedProviders = computed(() => {
                return availableProviders.value.some(provider => provider.selected);
            });

            const getTotalSelectedModels = () => {
                return availableProviders.value
                    .filter(p => p.selected)
                    .reduce((total, provider) => total + provider.selectedModels.length, 0);
            };

            const canProceed = computed(() => {
                return uploadedFile.value !== null;
            });

            const notificationSupported = computed(() => {
                return typeof window !== 'undefined' && 'Notification' in window;
            });

            const canStartAnnotation = computed(() => {
                if (!hasSelectedProviders.value) return false;
                const selectedProviders = availableProviders.value.filter(p => p.selected);
                return selectedProviders.every(p =>
                    p.apiKey.trim() !== '' && p.selectedModels.length > 0
                );
            });

            const selectedModelNames = computed(() => {
                return availableProviders.value
                    .filter(p => p.selected)
                    .flatMap(p => p.selectedModels.map(modelId =>
                        p.models.find(m => m.id === modelId)?.name || modelId
                    ));
            });

            const predefinedSpeciesValues = computed(() => {
                return predefinedSpeciesList.value.map(s => s.value);
            });

            const hasValidModelSelection = computed(() => {
                const selectedProviders = availableProviders.value.filter(p => p.selected);
                return selectedProviders.every(p => p.selectedModels.length > 0);
            });

            // API connection test state
            const hasApiKeysToTest = computed(() => {
                return availableProviders.value.some(provider =>
                    provider.selected && provider.apiKey.trim()
                );
            });

            const isAnyTesting = computed(() => {
                return availableProviders.value.some(provider => provider.testing);
            });

            const availableConsensusModels = computed(() => {
                const models = [];
                availableProviders.value
                    .filter(p => p.selected)
                    .forEach(provider => {
                        provider.selectedModels.forEach(modelId => {
                            const model = provider.models.find(m => m.id === modelId);
                            if (model) {
                                models.push({
                                    value: `${provider.id}:${modelId}`,
                                    label: `${provider.name} - ${model.name}`,
                                    provider: provider.id,
                                    modelId: modelId
                                });
                            }
                        });
                    });
                return models;
            });

            // Methods
            const showToast = (message, type = 'info', duration = 5000) => {
                const toast = {
                    id: Date.now() + Math.random(),
                    message,
                    type
                };
                toasts.value.push(toast);

                setTimeout(() => {
                    removeToast(toast.id);
                }, duration);
            };

            const removeToast = (id) => {
                const index = toasts.value.findIndex(toast => toast.id === id);
                if (index > -1) {
                    toasts.value.splice(index, 1);
                }
            };

            const getToastIcon = (type) => {
                switch (type) {
                    case 'success': return 'fas fa-check-circle';
                    case 'error': return 'fas fa-exclamation-circle';
                    case 'warning': return 'fas fa-exclamation-triangle';
                    default: return 'fas fa-info-circle';
                }
            };

            const toggleLanguage = () => {
                currentLang.value = currentLang.value === 'en' ? 'zh' : 'en';
                showToast(t.value('languageSwitched'), 'success');
            };

            const startAnnotation = () => {
                currentStep.value = 1;
            };

            const goBack = () => {
                if (currentStep.value > 0) {
                    currentStep.value--;
                }
            };

            const nextStep = () => {
                if (canProceed.value) {
                    currentStep.value++;
                }
            };

            // File handling
            const handleDragOver = (event) => {
                event.preventDefault();
                event.stopPropagation();
                isDragover.value = true;
            };

            const handleDragLeave = (event) => {
                // Only trigger if we're actually leaving the upload area
                if (!event.currentTarget.contains(event.relatedTarget)) {
                    isDragover.value = false;
                }
            };

            const handleFileDrop = (event) => {
                event.preventDefault();
                event.stopPropagation();
                isDragover.value = false;
                const files = event.dataTransfer.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            };

            const handleFileSelect = (event) => {
                const files = event.target.files;
                if (files.length > 0) {
                    handleFile(files[0]);
                }
            };

            // Handle click on upload area
            const handleUploadAreaClick = (event) => {
                // Don't trigger if clicking on button or if file is already uploaded
                if (event.target.closest('button') || uploadedFile.value) {
                    return;
                }
                // Trigger file input click
                const fileInput = document.querySelector('input[accept=".csv,.tsv,.xlsx"]');
                if (fileInput) {
                    fileInput.click();
                }
            };

            const handleFile = async (file) => {
                // Validate file type
                const validExtensions = ['.csv', '.tsv', '.xlsx'];
                const fileExt = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();

                if (!validExtensions.includes(fileExt)) {
                    uploadError.value = t.value('errorUnsupportedFileType').replace('{ext}', fileExt);
                    return;
                }

                // Validate file size (16MB limit)
                if (file.size > 16 * 1024 * 1024) {
                    uploadError.value = t.value('errorFileTooLarge');
                    return;
                }

                isUploading.value = true;
                uploadError.value = '';

                try {
                    const formData = new FormData();
                    formData.append('file', file);

                    const response = await apiRequest('/api/upload', {
                        method: 'POST',
                        body: formData
                    });

                    uploadedFile.value = file;
                    taskId.value = response.data.task_id;
                    dataPreview.value = response.data.file_info?.preview || [];
                    dataColumns.value = response.data.file_info?.columns || [];

                    showToast(t.value('uploadSuccess'), 'success');

                } catch (error) {
                    console.error('Upload error:', error);
                    uploadError.value = error.response?.data?.error || t.value('errorUploadFailed');
                    showToast(t.value('uploadFailed'), 'error');
                } finally {
                    isUploading.value = false;
                }
            };

            const removeFile = () => {
                uploadedFile.value = null;
                taskId.value = null;
                dataPreview.value = [];
                dataColumns.value = [];
                uploadError.value = '';
            };

            const formatFileSize = (bytes) => {
                if (bytes === 0) return '0 Bytes';
                const k = 1024;
                const sizes = ['Bytes', 'KB', 'MB', 'GB'];
                const i = Math.floor(Math.log(bytes) / Math.log(k));
                return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
            };

            const downloadSample = () => {
                window.location.href = '/api/sample';
            };

            // .env file handling
            const handleEnvFileUpload = async (event) => {
                const file = event.target.files[0];
                if (!file) return;

                // Clear previous message
                envImportMessage.value = null;

                try {
                    const content = await file.text();
                    const lines = content.split('\n');
                    const apiKeys = {};
                    let successCount = 0;
                    let failCount = 0;

                    // Parse .env file
                    lines.forEach(line => {
                        const trimmedLine = line.trim();
                        if (trimmedLine && !trimmedLine.startsWith('#')) {
                            const [key, ...valueParts] = trimmedLine.split('=');
                            const value = valueParts.join('=').trim();

                            if (key && value) {
                                // Remove quotes if present
                                const cleanValue = value.replace(/^["']|["']$/g, '');

                                // Map env keys to provider IDs
                                const keyMapping = {
                                    'OPENAI_API_KEY': 'openai',
                                    'ANTHROPIC_API_KEY': 'anthropic',
                                    'GEMINI_API_KEY': 'gemini',
                                    'GOOGLE_API_KEY': 'gemini', // Alternative name
                                    'GROK_API_KEY': 'grok',
                                    'XAI_API_KEY': 'grok', // Alternative name
                                    'DEEPSEEK_API_KEY': 'deepseek',
                                    'MOONSHOT_API_KEY': 'kimi',
                                    'KIMI_API_KEY': 'kimi',
                                    'QWEN_API_KEY': 'qwen',
                                    'ZHIPU_API_KEY': 'zhipu',
                                    'GLM_API_KEY': 'zhipu', // Alternative name
                                    'STEPFUN_API_KEY': 'stepfun',
                                    'MINIMAX_API_KEY': 'minimax',
                                    'OPENROUTER_API_KEY': 'openrouter'
                                };

                                const providerId = keyMapping[key.trim()];
                                if (providerId) {
                                    apiKeys[providerId] = cleanValue;
                                }
                            }
                        }
                    });

                    // Apply API keys to providers
                    Object.entries(apiKeys).forEach(([providerId, apiKey]) => {
                        const provider = availableProviders.value.find(p => p.id === providerId);
                        if (provider) {
                            provider.apiKey = apiKey;
                            provider.selected = true; // Auto-select provider when API key is imported
                            successCount++;
                        } else {
                            failCount++;
                        }
                    });

                    // Show result message
                    if (successCount > 0 && failCount === 0) {
                        envImportMessage.value = {
                            type: 'success',
                            text: t.value('envImportSuccess').replace('{count}', successCount)
                        };
                    } else if (successCount > 0 && failCount > 0) {
                        envImportMessage.value = {
                            type: 'warning',
                            text: t.value('envImportPartial')
                                .replace('{success}', successCount)
                                .replace('{failed}', failCount)
                        };
                    } else {
                        envImportMessage.value = {
                            type: 'error',
                            text: t.value('envImportError')
                        };
                    }

                    // Clear message after 5 seconds
                    setTimeout(() => {
                        envImportMessage.value = null;
                    }, 5000);

                } catch (error) {
                    console.error('Error reading .env file:', error);
                    envImportMessage.value = {
                        type: 'error',
                        text: t.value('envImportError')
                    };
                }

                // Reset file input
                event.target.value = '';
            };

            // .env file drag and drop handlers
            const handleEnvDragOver = (event) => {
                event.preventDefault();
                event.stopPropagation();
                isEnvDragover.value = true;
            };

            const handleEnvDragLeave = (event) => {
                // Only trigger if we're actually leaving the banner area
                if (!event.currentTarget.contains(event.relatedTarget)) {
                    isEnvDragover.value = false;
                }
            };

            const handleEnvFileDrop = async (event) => {
                event.preventDefault();
                event.stopPropagation();
                isEnvDragover.value = false;

                const files = event.dataTransfer.files;
                if (files.length === 0) return;

                const file = files[0];

                // Check if it's a .env file
                if (!file.name.endsWith('.env') && file.name !== '.env') {
                    envImportMessage.value = {
                        type: 'error',
                        text: t.value('envFileTypeError')
                    };
                    setTimeout(() => {
                        envImportMessage.value = null;
                    }, 5000);
                    return;
                }

                // Process the file using the existing handler
                handleEnvFileUpload({ target: { files: [file] } });
            };

            // Tissue selection methods
            const selectPredefinedTissue = (selectedTissue) => {
                tissue.value = selectedTissue;
                showCustomInput.value = false;
            };

            const clearTissue = () => {
                tissue.value = '';
                showCustomInput.value = false;
            };

            // Species selection methods
            const selectPredefinedSpecies = (selectedSpecies) => {
                species.value = selectedSpecies;
                showCustomSpeciesInput.value = false;
            };

            const clearSpecies = () => {
                species.value = '';
                showCustomSpeciesInput.value = false;
            };

            // Provider management
            const toggleProvider = (provider) => {
                provider.selected = !provider.selected;
            };

            const addCustomModel = (provider) => {
                const rawModelId = provider.customModel.trim();
                const providerPrefix = `${provider.id}:`;
                const modelId = rawModelId.startsWith(providerPrefix)
                    ? rawModelId.slice(providerPrefix.length)
                    : rawModelId;
                if (!modelId) {
                    showToast(t.value('customModelRequired'), 'warning');
                    return;
                }
                if (modelId.length > 300) {
                    showToast(t.value('customModelTooLong'), 'warning');
                    return;
                }
                if (!provider.models.some(model => model.id === modelId)) {
                    provider.models.push({ id: modelId, name: modelId });
                }
                if (!provider.selectedModels.includes(modelId)) {
                    provider.selectedModels.push(modelId);
                }
                provider.customModel = '';
            };

            const toggleApiKeyVisibility = (provider) => {
                provider.showApiKey = !provider.showApiKey;
            };

            // API Key Testing
            const testApiKey = async (provider) => {
                if (!provider.apiKey.trim()) {
                    showToast(t.value('apiKeyRequired'), 'warning');
                    return;
                }

                // Find the provider in reactive state.
                const providerIndex = availableProviders.value.findIndex(p => p.id === provider.id);
                if (providerIndex === -1) return;

                // Update the reactive provider entry.
                availableProviders.value[providerIndex].testing = true;
                availableProviders.value[providerIndex].testResult = null;

                try {
                    // Let backend pick the cost-effective test model
                    const response = await apiRequest('/api/test-api-key', {
                        method: 'POST',
                        json: {
                            provider: provider.id,
                            api_key: provider.apiKey,
                            model: provider.selectedModels[0] || provider.models[0]?.id
                        }
                    });

                    // Validate the response contract.
                    if (response.data && response.data.valid) {
                        availableProviders.value[providerIndex].testResult = {
                            status: 'success',
                            message: response.data.message || t.value('apiKeyValid')
                        };
                        availableProviders.value[providerIndex].lastTested = new Date();

                        showToast(`${provider.name} ${t.value('apiKeyTestSuccess')}`, 'success');
                    } else {
                        // Treat an unexpected false result as a failed validation.
                        throw new Error(response.data.error || t.value('apiKeyTestFailed'));
                    }

                } catch (error) {
                    console.error('API test error:', error);
                    const rawError = error.response?.data?.error || error.message || t.value('apiKeyTestFailed');
                    const friendlyError = getFriendlyErrorMessage(rawError, provider);

                    availableProviders.value[providerIndex].testResult = {
                        status: 'error',
                        message: friendlyError
                    };

                    showToast(friendlyError, 'error');
                } finally {
                    // Always leave the loading state.
                    availableProviders.value[providerIndex].testing = false;
                }
            };

            // Test selected API keys sequentially to avoid request bursts.
            const testAllApiKeys = async () => {
                const selectedProviders = availableProviders.value.filter(p =>
                    p.selected && p.apiKey.trim()
                );

                if (selectedProviders.length === 0) {
                    showToast(t.value('noApiKeysToTest'), 'warning');
                    return;
                }

                for (const provider of selectedProviders) {
                    await testApiKey(provider);
                    // Keep a short delay between providers.
                    await new Promise(resolve => setTimeout(resolve, 500));
                }
            };

            // Format the relative test time.
            const formatTestTime = (date) => {
                const now = new Date();
                const diff = Math.floor((now - date) / 1000);

                if (diff < 60) return t.value('justNow');
                if (diff < 3600) return t.value('minutesAgo', { minutes: Math.floor(diff / 60) });
                return t.value('hoursAgo', { hours: Math.floor(diff / 3600) });
            };

            // Map common API errors to actionable messages.
            const getFriendlyErrorMessage = (error, provider) => {
                const errorMap = {
                    'Invalid API key': t.value('errorInvalidApiKey', { provider: provider.name }),
                    'Rate limit exceeded': t.value('errorRateLimit', { provider: provider.name }),
                    'Insufficient permissions': t.value('errorInsufficientPermissions', { provider: provider.name }),
                    'Request timeout': t.value('errorTimeout', { provider: provider.name }),
                    'Connection error': t.value('errorConnection', { provider: provider.name }),
                    'Quota exceeded': t.value('errorQuotaExceeded', { provider: provider.name })
                };

                // Match known error categories.
                for (const [key, message] of Object.entries(errorMap)) {
                    if (error.toLowerCase().includes(key.toLowerCase())) {
                        return message;
                    }
                }

                // Preserve the provider error when no category matches.
                return t.value('errorGeneric', { provider: provider.name, error });
            };

            // Clear stale validation state when an API key changes.
            const onApiKeyChange = (provider) => {
                const providerIndex = availableProviders.value.findIndex(p => p.id === provider.id);
                if (providerIndex === -1) return;

                if (availableProviders.value[providerIndex].testResult) {
                    availableProviders.value[providerIndex].testResult = null;
                }
            };

            const getProviderIcon = (providerId) => {
                const icons = {
                    'openai': 'fas fa-robot',  // Changed from fab fa-openai which might not exist
                    'anthropic': 'fas fa-comment-dots',
                    'gemini': 'fab fa-google',
                    'grok': 'fab fa-twitter',
                    'deepseek': 'fas fa-brain',
                    'kimi': 'fas fa-moon',
                    'qwen': 'fas fa-cloud',
                    'zhipu': 'fas fa-microchip',
                    'stepfun': 'fas fa-stairs',
                    'minimax': 'fas fa-expand-arrows-alt',
                    'openrouter': 'fas fa-route'
                };
                return icons[providerId] || 'fas fa-robot';
            };

            const getProviderIconClass = (providerId) => {
                const classes = {
                    'openai': 'openai',
                    'anthropic': 'anthropic',
                    'gemini': 'gemini',
                    'grok': 'grok',
                    'deepseek': 'deepseek',
                    'kimi': 'kimi',
                    'qwen': 'qwen',
                    'zhipu': 'zhipu',
                    'stepfun': 'stepfun',
                    'minimax': 'minimax',
                    'openrouter': 'openrouter'
                };
                return classes[providerId] || 'default';
            };

            // Provider API documentation URLs
            const providerApiUrls = {
                'openai': 'https://platform.openai.com/settings/organization/api-keys',
                'anthropic': 'https://console.anthropic.com/settings/keys',
                'gemini': 'https://aistudio.google.com/app/apikey',
                'grok': 'https://console.x.ai/',
                'deepseek': 'https://platform.deepseek.com/api_keys',
                'kimi': 'https://platform.moonshot.cn/console/api-keys',
                'qwen': 'https://dashscope.console.aliyun.com/apiKey',
                'zhipu': 'https://open.bigmodel.cn/usercenter/apikeys',
                'stepfun': 'https://platform.stepfun.com/interface/key',
                'minimax': 'https://platform.minimaxi.com/user-center/basic-information/interface-key',
                'openrouter': 'https://openrouter.ai/settings/keys'
            };

            const getProviderApiUrl = (providerId) => {
                return providerApiUrls[providerId] || null;
            };

            // Processing
            const startProcessing = async () => {
                if (!canStartAnnotation.value) return;

                currentStep.value = 3;
                progressPercent.value = 0;
                processingError.value = '';
                notificationSent.value = false;
                completedProcessing.value = false;

                const selectedProviders = availableProviders.value.filter(p => p.selected);
                const payload = {
                    task_id: taskId.value,
                    species: species.value,
                    tissue: tissue.value,
                    models: selectedProviders.flatMap(p =>
                        p.selectedModels.map(modelId => `${p.id}:${modelId}`)
                    ),
                    api_keys: Object.fromEntries(selectedProviders.map(p => [p.id, p.apiKey])),
                    consensusThreshold: consensusThreshold.value,
                    entropyThreshold: entropyThreshold.value,
                    maxDiscussionRounds: maxDiscussionRounds.value,
                    consensusModel: consensusModel.value || null
                };

                try {
                    await apiRequest('/api/annotate', { method: 'POST', json: payload });
                    taskStatus.value = 'Queued';
                    startPolling();
                    startProcessingTimer();
                    showToast(t.value('analysisStarted'), 'success');
                } catch (error) {
                    console.error('Annotation error:', error);
                    // Return to Step 2 so user can adjust settings and retry;
                    // all configuration (API keys, models, parameters) is preserved.
                    currentStep.value = 2;
                    const errorMsg = error.response?.data?.error || t.value('errorStartAnalysis');
                    showToast(errorMsg, 'error');
                }
            };

            const startPolling = () => {
                if (pollTimer) clearTimeout(pollTimer);
                let pollErrorCount = 0;
                let resultErrorCount = 0;
                let pollInterval = 2000; // Start at 2s

                const schedulePoll = () => {
                    pollTimer = setTimeout(pollOnce, pollInterval);
                };

                const pollOnce = async () => {
                    try {
                        const response = await apiRequest(`/api/tasks/${taskId.value}`);
                        // Success — reset backoff
                        pollErrorCount = 0;
                        pollInterval = 2000;

                        const status = response.data.status;
                        taskStatus.value = formatStatus(status);

                        // Update progress from backend
                        if (response.data.progress !== undefined) {
                            progressPercent.value = response.data.progress;
                        }

                        // Update detailed progress if available
                        if (response.data.progress_details) {
                            progressDetails.value = response.data.progress_details;
                        }

                        if (status === 'completed') {
                            clearInterval(processingTimer);

                            if (!completedProcessing.value) {
                                progressPercent.value = 100;
                                const loaded = await getResults();
                                if (!loaded) {
                                    resultErrorCount += 1;
                                    if (resultErrorCount >= 5) {
                                        showToast(t.value('getResultsFailed'), 'error');
                                        return;
                                    }
                                    pollInterval = Math.min(30000, 2000 * Math.pow(2, resultErrorCount));
                                    schedulePoll();
                                    return;
                                }

                                completedProcessing.value = true;
                                if (response.data.persistence_failed) {
                                    showToast(t.value('persistenceWarning'), 'warning');
                                } else {
                                    showToast(t.value('analysisCompleted'), 'success');
                                }

                                if (!notificationSent.value) {
                                    notificationSent.value = true;
                                    const clusterCount = Object.keys(results.value.consensus || {}).length;
                                    const filename = uploadedFile.value?.name || 'annotation';
                                    sendNotification(
                                        t.value('notificationTitle'),
                                        t.value('notificationBody')
                                            .replace('{filename}', filename)
                                            .replace('{count}', clusterCount)
                                    );
                                }
                            }
                            return; // Terminal — stop polling
                        } else if (status === 'failed') {
                            processingError.value = response.data.error || t.value('errorProcessingFailed');
                            clearInterval(processingTimer);
                            showToast(t.value('analysisFailed'), 'error');
                            return;
                        } else if (status === 'timeout') {
                            processingError.value = response.data.error || t.value('errorProcessingTimeout');
                            clearInterval(processingTimer);
                            showToast(t.value('analysisTimeout'), 'warning');
                            return;
                        } else if (status === 'cancelled') {
                            processingError.value = response.data.error || t.value('errorTaskCancelled');
                            clearInterval(processingTimer);
                            showToast(t.value('analysisCancelled'), 'warning');
                            return;
                        }

                        // Non-terminal status — keep polling
                        schedulePoll();
                    } catch (error) {
                        const status = error.response?.status;
                        if (status && status >= 400 && status < 500) {
                            // 4xx = permanent error (task gone / invalid).
                            // Stop polling and tell the user.
                            console.error('Task polling stopped (permanent error):', status);
                            processingError.value = error.response?.data?.error || t.value('errorProcessingFailed');
                            clearInterval(processingTimer);
                            showToast(t.value('errorProcessingFailed'), 'error');
                            return;
                        }
                        // 5xx or network error — transient, keep retrying
                        console.error('Status check error (transient):', error);
                        pollErrorCount++;
                        pollInterval = Math.min(30000, 2000 * Math.pow(2, pollErrorCount));
                        schedulePoll();
                    }
                };

                schedulePoll();
            };

            const startProcessingTimer = () => {
                processingTime.value = 0;
                processingTimer = setInterval(() => {
                    processingTime.value++;
                }, 1000);
            };

            const formatStatus = (status) => {
                const keyMap = {
                    'file_ready': 'statusFileReady',
                    'queued': 'statusQueued',
                    'processing': 'statusProcessing',
                    'completed': 'statusCompleted',
                    'failed': 'statusFailed',
                    'timeout': 'statusTimeout',
                    'cancelled': 'statusCancelled'
                };
                const key = keyMap[status];
                return key ? t.value(key) : status;
            };

            const getProgressPhaseClass = (phase) => {
                const phaseClasses = {
                    'annotation': 'progress-annotation',
                    'consensus': 'progress-consensus',
                    'controversy': 'progress-controversy',
                    'starting': 'progress-starting',
                    'processing': 'progress-processing'
                };
                return phaseClasses[phase] || 'progress-default';
            };

            const getProgressPhaseLabel = (phase) => {
                const keyMap = {
                    'annotation': 'modelAnnotation',
                    'consensus': 'consensusChecking',
                    'controversy': 'controversyResolution',
                    'starting': 'phaseStarting',
                    'processing': 'statusProcessing'
                };
                const key = keyMap[phase];
                return key ? t.value(key) : phase;
            };

            const getResults = async () => {
                try {
                    const response = await apiRequest(`/api/results/${taskId.value}`);
                    if (response.data.error) {
                        processingError.value = response.data.error;
                        return false;
                    } else if (response.data.task_id) {
                        results.value = response.data;
                        currentStep.value = 4;
                        processingError.value = '';
                        return true;
                    } else {
                        processingError.value = t.value('errorInvalidResultsFormat');
                        return false;
                    }
                } catch (error) {
                    console.error('Results error:', error);
                    processingError.value = t.value('errorGetResults');
                    return false;
                }
            };

            // Results handling
            const getConfidenceClass = (consensus) => {
                if (consensus >= 0.8) return 'high';
                if (consensus >= 0.6) return 'medium';
                return 'low';
            };

            const getConfidenceLabel = (consensus) => {
                if (consensus >= 0.8) return t.value('confidenceHigh');
                if (consensus >= 0.6) return t.value('confidenceMedium');
                return t.value('confidenceLow');
            };

            const formatDuration = (seconds) => {
                const mins = Math.floor(seconds / 60);
                const secs = seconds % 60;
                return `${mins}:${secs.toString().padStart(2, '0')}`;
            };

            const fetchDownload = async (url) => {
                const response = await fetch(url);
                if (!response.ok) {
                    let msg = t.value('downloadFailed');
                    try {
                        const body = await response.json();
                        if (body.error) msg = body.error;
                    } catch (_) { /* not JSON */ }
                    showToast(msg, 'error');
                    return;
                }
                const disposition = response.headers.get('Content-Disposition') || '';
                const match = disposition.match(/filename="?([^"]+)"?/);
                const filename = match ? match[1] : url.split('/').pop();
                const blob = await response.blob();
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = filename;
                a.click();
                URL.revokeObjectURL(a.href);
            };

            const exportResults = (format) => {
                if (!taskId.value) return;
                fetchDownload(`/api/download/${taskId.value}/${format}`);
            };

            const downloadAllResults = () => {
                exportResults('excel');
            };

            const downloadAnnotationLogs = () => {
                if (!taskId.value) return;
                fetchDownload(`/api/download-logs/${taskId.value}`);
            };

            const downloadDiscussionDetails = () => {
                if (!taskId.value) return;
                fetchDownload(`/api/download-discussion/${taskId.value}`);
            };

            const startNewAnnotation = () => {
                // Reset all state
                currentStep.value = 1;  // Go directly to upload step instead of home
                uploadedFile.value = null;
                dataPreview.value = [];
                dataColumns.value = [];
                results.value = {};
                uploadError.value = '';
                processingError.value = '';
                taskId.value = null;
                progressPercent.value = 0;
                processingTime.value = 0;
                notificationSent.value = false;
                completedProcessing.value = false;

                // Clear timers
                if (pollTimer) clearTimeout(pollTimer);
                if (processingTimer) clearInterval(processingTimer);

                // Reset providers API keys for security
                availableProviders.value.forEach(provider => {
                    provider.apiKey = '';
                    provider.showApiKey = false;
                });

                showToast(t.value('readyForNewUpload'), 'success');
            };

            // Rerun with different parameters
            const rerunWithDifferentParams = async () => {
                // Reset backend state so /api/annotate accepts the task again
                try {
                    await apiRequest(`/api/tasks/${taskId.value}/reset`, { method: 'POST' });
                } catch (error) {
                    showToast(error.response?.data?.error || t.value('errorResetTask'), 'error');
                    return;
                }

                // Reset frontend processing state (keep file + config)
                currentStep.value = 2;  // Return to configuration step
                results.value = {};
                processingError.value = '';
                progressPercent.value = 0;
                processingTime.value = 0;
                progressDetails.value = null;
                taskStatus.value = 'Preparing';
                notificationSent.value = false;
                completedProcessing.value = false;

                // Clear timers
                if (pollTimer) clearTimeout(pollTimer);
                if (processingTimer) clearInterval(processingTimer);

                showToast(t.value('readyToRerun'), 'success');
            };

            // Notification functions
            const requestNotificationPermission = async () => {
                if ('Notification' in window && Notification.permission === 'default') {
                    const permission = await Notification.requestPermission();
                    notificationPermission.value = permission;
                    if (permission === 'granted') {
                        showToast(t.value('notificationsEnabled'), 'success');
                    } else if (permission === 'denied') {
                        showToast(t.value('notificationsDenied'), 'warning');
                    }
                }
            };

            const sendNotification = (title, body) => {
                if ('Notification' in window && Notification.permission === 'granted') {
                    new Notification(title, { body });
                }
            };

            const loadProviderDefaults = async () => {
                try {
                    const response = await apiRequest('/api/provider-catalog');
                    const defaults = response.data.defaults || {};
                    const modelCatalog = response.data.models || {};
                    const providerNames = response.data.provider_names || {};
                    const supportedProviders = response.data.providers || Object.keys(defaults);
                    const previousProviders = new Map(
                        availableProviders.value.map(provider => [provider.id, provider])
                    );
                    availableProviders.value = supportedProviders.map(providerId => {
                        const existing = previousProviders.get(providerId);
                        if (existing) {
                            existing.name = providerNames[providerId] || existing.name;
                            return existing;
                        }
                        return createProviderState(
                            providerId,
                            providerNames[providerId] || providerId
                        );
                    });
                    availableProviders.value.forEach(provider => {
                        const providerModels = modelCatalog[provider.id];
                        if (Array.isArray(providerModels)) {
                            provider.models = providerModels.filter(model =>
                                model && typeof model.id === 'string' && typeof model.name === 'string'
                            );
                        }
                        const defaultModel = defaults[provider.id];
                        if (!defaultModel) return;
                        if (!provider.models.some(model => model.id === defaultModel)) {
                            provider.models.unshift({ id: defaultModel, name: defaultModel });
                        }
                        if (provider.selectedModels.length === 0) {
                            provider.selectedModels = [defaultModel];
                        }
                    });
                } catch (error) {
                    console.error('Failed to load provider defaults:', error);
                }
            };

            // Lifecycle
            onMounted(() => {
                loadProviderDefaults();

                // Fetch and display deployment info
                fetch('/api/deployment-info')
                    .then(response => response.json())
                    .then(data => {
                        if (data.deploy_time) {
                            const deployTime = new Date(data.deploy_time);
                            const options = {
                                year: 'numeric',
                                month: 'long',
                                day: 'numeric',
                                hour: '2-digit',
                                minute: '2-digit',
                                timeZoneName: 'short'
                            };
                            const formattedTime = deployTime.toLocaleString('en-US', options);
                            document.getElementById('deployTime').textContent = formattedTime;
                        }
                    })
                    .catch(error => {
                        console.error('Failed to fetch deployment info:', error);
                        document.getElementById('deployTime').textContent = 'Unknown';
                    });
            });

            onBeforeUnmount(() => {
                if (pollTimer) clearTimeout(pollTimer);
                if (processingTimer) clearInterval(processingTimer);
            });

            return {
                // State
                currentLang,
                currentStep,
                globalLoading,
                loadingMessage,
                toasts,
                isDragover,
                isUploading,
                uploadError,
                uploadedFile,
                dataPreview,
                dataColumns,
                species,
                tissue,
                consensusThreshold,
                entropyThreshold,
                maxDiscussionRounds,
                consensusModel,
                availableConsensusModels,
                showCustomInput,
                showCustomSpeciesInput,
                predefinedSpeciesList,
                availableProviders,
                taskStatus,
                processingError,
                progressPercent,
                processingTime,
                progressDetails,
                results,
                envImportMessage,
                isEnvDragover,
                notificationPermission,

                // Computed
                t,
                hasSelectedProviders,
                getTotalSelectedModels,
                canProceed,
                notificationSupported,
                canStartAnnotation,
                selectedModelNames,
                predefinedSpeciesValues,
                hasValidModelSelection,
                hasApiKeysToTest,
                isAnyTesting,
                filteredTissuesList,

                // Methods
                removeToast,
                getToastIcon,
                toggleLanguage,
                startAnnotation,
                goBack,
                nextStep,
                handleDragOver,
                handleDragLeave,
                handleFileDrop,
                handleFileSelect,
                handleUploadAreaClick,
                removeFile,
                formatFileSize,
                downloadSample,
                handleEnvFileUpload,
                handleEnvDragOver,
                handleEnvDragLeave,
                handleEnvFileDrop,
                selectPredefinedTissue,
                clearTissue,
                selectPredefinedSpecies,
                clearSpecies,
                toggleProvider,
                addCustomModel,
                toggleApiKeyVisibility,
                testApiKey,
                testAllApiKeys,
                formatTestTime,
                onApiKeyChange,
                getProgressPhaseClass,
                getProgressPhaseLabel,
                getProviderIcon,
                getProviderIconClass,
                getProviderApiUrl,
                startProcessing,
                getConfidenceClass,
                getConfidenceLabel,
                formatDuration,
                exportResults,
                downloadAllResults,
                downloadAnnotationLogs,
                downloadDiscussionDetails,
                startNewAnnotation,
                rerunWithDifferentParams,
                requestNotificationPermission
            };
        }
    };

    // Initialize the app with error handling
    try {
        const app = createApp(App);
        app.mount('#app');
        appInitialized = true;
        document.getElementById('appLoadError')?.remove();
    } catch (error) {
        console.error('Failed to initialize Vue app:', error);
        showAppLoadError('The application failed to initialize. Refresh the page to try again.');
    }
}

window.initializeApp();
