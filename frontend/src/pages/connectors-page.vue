<script setup lang="ts">
import { 
    Activity, 
    ArrowRight, 
    Bot, 
    Brain, 
    Cpu, 
    Database, 
    Globe, 
    HardDrive, 
    Network, 
    Radio, 
    Router, 
    ShieldCheck, 
    Video, 
    Wifi, 
    Zap,
    Plus,
    X,
    Settings,
    Save,
    Trash2,
    LayoutDashboard,
    Workflow,
    MousePointer2
} from 'lucide-vue-next';
import { ref, reactive, computed } from 'vue';
import { toast } from 'vue-sonner';

// State
const viewMode = ref<'dashboard' | 'workflow'>('dashboard');
const connecting = ref(true)
setTimeout(() => {
    connecting.value = false
}, 2000)

// Data Models
interface Sensor {
    id: number;
    name: string;
    type: 'video' | 'iot' | 'power' | 'network';
    status: 'active' | 'warning' | 'error' | 'connecting';
    stats: string;
    icon: any;
}

const sensors = ref<Sensor[]>([
    { id: 1, name: 'Traffic Cam #802', type: 'video', status: 'active', stats: '4.2 Mbps', icon: Video },
    { id: 2, name: 'Air Quality Stn A', type: 'iot', status: 'active', stats: '98% Signal', icon: Wifi },
    { id: 3, name: 'Grid Relay South', type: 'power', status: 'warning', stats: 'Load 89%', icon: Zap },
    { id: 4, name: 'Metro Gateway', type: 'network', status: 'active', stats: '12ms Latency', icon: Router },
]);

const streamConfig = reactive({
    anomalyDetection: true,
    schemaValidation: true,
    autoCleaning: true,
    throughputLimit: 5, // GB/s
    encryptionLevel: 'Standard'
});

// Modals
const showAddModal = ref(false);
const showConfigModal = ref(false);

const newSourceForm = reactive({
    name: '',
    type: 'iot' as 'video' | 'iot' | 'power' | 'network',
    endpoint: ''
});

// Logs Simulation
const logs = ref<string[]>([
   "[SYSTEM] Flowmatic Engine v2.1 initialized",
   "[INFO] Connected to federated cluster south-1",
   "[STREAM] Ingestion rate stable at 4.1 GB/s", 
]);

// Actions
function addSource() {
    if (!newSourceForm.name) {
        toast.error('Source name is required');
        return;
    }

    const icons = {
        video: Video,
        iot: Wifi,
        power: Zap,
        network: Router
    };

    const newSensor: Sensor = {
        id: Date.now(),
        name: newSourceForm.name,
        type: newSourceForm.type,
        status: 'connecting',
        stats: 'Initializing...',
        icon: icons[newSourceForm.type]
    };

    sensors.value.push(newSensor);
    showAddModal.value = false;
    newSourceForm.name = '';

    toast.message('Connecting to Source', { description: `Handshaking with ${newSensor.name}...` });

    // Simulate connection
    setTimeout(() => {
        const s = sensors.value.find(x => x.id === newSensor.id);
        if (s) {
            s.status = 'active';
            s.stats = 'Live Stream';
            logs.value.unshift(`[INFO] New source connected: ${newSensor.name} (${newSensor.type})`);
            toast.success('Source Connected');
        }
    }, 1500);
}

function removeSensor(id: number) {
    const sensor = sensors.value.find(s => s.id === id);
    if (!sensor) return;
    
    // Simulate removing
    toast('Disconnecting Source', {
        action: {
            label: 'Confirm',
            onClick: () => {
                sensors.value = sensors.value.filter(s => s.id !== id);
                logs.value.unshift(`[WARN] Source disconnected: ${sensor.name}`);
                toast.success('Source Removed');
            }
        }
    });
}

function saveConfiguration() {
    // Simulate saving settings
    showConfigModal.value = false;
    logs.value.unshift(`[CONFIG] Stream parameters updated: Limit=${streamConfig.throughputLimit}GB/s`);
    toast.success('Configuration Applied', { description: 'Stream processing rules updated successfully.' });
}

function openDataDestination(name: string) {
    toast.message(`${name} Interface`, { description: 'Secure channel established. Opening visualization...' });
}

// Helper to get formatted icon
const getTypeIcon = (type: string) => {
    switch(type) {
        case 'video': return Video;
        case 'iot': return Wifi;
        case 'power': return Zap;
        case 'network': return Router;
        default: return Radio;
    }
}
</script>

<template>
    <div class="min-h-screen p-6 lg:p-10 space-y-8 relative">
        <!-- Header -->
        <div class="flex items-center justify-between">
            <div class="space-y-1">
                <div class="flex items-center gap-3">
                    <h1 class="text-2xl font-bold text-foreground">Connectors & Streams</h1>
                    <span class="px-2 py-0.5 rounded-full bg-primary/10 text-primary text-xs font-bold uppercase tracking-wider border border-primary/20">
                        Beta Preview
                    </span>
                </div>
                <p class="text-foreground/60 max-w-2xl">
                    Real-time ingestion and processing for high-frequency Smart City sensor data. 
                    Configure federated learning nodes and automated data routing.
                </p>
            </div>

            <!-- View Toggle -->
            <div class="flex items-center bg-card border border-border rounded-lg p-1 shadow-sm">
                <button 
                    @click="viewMode = 'dashboard'"
                    :class="[
                        'px-3 py-1.5 rounded-md text-sm font-medium flex items-center gap-2 transition-all',
                        viewMode === 'dashboard' 
                            ? 'bg-primary text-primary-foreground shadow-sm' 
                            : 'text-foreground/60 hover:text-foreground hover:bg-surface-2'
                    ]"
                >
                    <LayoutDashboard class="w-4 h-4" />
                    Dashboard
                </button>
                <button 
                    @click="viewMode = 'workflow'"
                    :class="[
                        'px-3 py-1.5 rounded-md text-sm font-medium flex items-center gap-2 transition-all',
                        viewMode === 'workflow' 
                            ? 'bg-primary text-primary-foreground shadow-sm' 
                            : 'text-foreground/60 hover:text-foreground hover:bg-surface-2'
                    ]"
                >
                    <Workflow class="w-4 h-4" />
                    Canvas
                </button>
            </div>
        </div>

        <!-- Notification Banner -->
        <div v-if="viewMode === 'dashboard'" class="bg-gradient-to-r from-indigo-500/10 via-purple-500/10 to-pink-500/10 border border-indigo-500/20 rounded-xl p-4 flex items-start gap-4">
            <div class="p-2 bg-indigo-500/20 rounded-lg">
                <Bot class="w-5 h-5 text-indigo-400" />
            </div>
            <div>
                <h3 class="text-sm font-bold text-foreground mb-1">Interactive Simulation Mode</h3>
                <p class="text-sm text-foreground/70 leading-relaxed">
                    You can now add data sources, configure the processing engine stream, and simulate network conditions. 
                    Changes are reflected in the terminal log below.
                </p>
            </div>
        </div>

        <!-- WORKFLOW CANVAS VIEW -->
        <div v-if="viewMode === 'workflow'" class="h-[600px] border border-border rounded-xl bg-[#0f0f13] relative overflow-hidden flex flex-col shadow-2xl animate-in fade-in zoom-in duration-300">
            <!-- Toolbar -->
            <div class="border-b border-border bg-card/50 px-4 py-2 flex items-center justify-between z-10 backdrop-blur-md">
                <div class="flex items-center gap-4 text-xs font-mono text-foreground/60">
                    <div class="flex items-center gap-1.5"><MousePointer2 class="w-3.5 h-3.5" /> Drag & Drop Enabled (Simulation)</div>
                    <div class="h-4 w-px bg-border"></div>
                    <div>Zoom: 100%</div>
                </div>
                <div class="flex items-center gap-2">
                    <button class="px-2 py-1 hover:bg-white/5 rounded text-xs text-foreground/70 transition-colors">Reset Layout</button>
                    <button class="px-2 py-1 bg-primary/20 text-primary border border-primary/20 rounded text-xs hover:bg-primary/30 transition-colors">Auto-Arrange</button>
                </div>
            </div>

            <div class="flex-1 relative overflow-auto cursor-grab active:cursor-grabbing">
                <!-- Grid Background -->
                <div class="absolute inset-0 z-0 pointer-events-none" style="background-image: radial-gradient(#333 1px, transparent 1px); background-size: 20px 20px; opacity: 0.3;"></div>

                 <!-- Connection Lines Layer (Static Visualization for Demo) -->
                <svg class="absolute inset-0 w-full h-full pointer-events-none z-0 overflow-visible opacity-50">
                    <defs>
                        <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                            <polygon points="0 0, 10 3.5, 0 7" fill="#6b7280" />
                        </marker>
                    </defs>
                    <!-- Simple Bezier Curves representing flow -->
                    <!-- Source Group to Processor -->
                    <path d="M 300 120 C 450 120, 450 300, 600 300" stroke="#6b7280" stroke-width="2" fill="none" marker-end="url(#arrowhead)" stroke-dasharray="5,5" class="animate-[dash_30s_linear_infinite]" />
                    <path d="M 300 220 C 450 220, 450 300, 600 300" stroke="#6b7280" stroke-width="2" fill="none" marker-end="url(#arrowhead)" />
                     <path d="M 300 320 C 450 320, 450 300, 600 300" stroke="#6b7280" stroke-width="2" fill="none" marker-end="url(#arrowhead)" />
                    
                    <!-- Processor to Destinations -->
                    <path d="M 850 300 C 950 300, 950 200, 1050 200" stroke="#6366f1" stroke-width="2" fill="none" marker-end="url(#arrowhead)" />
                    <path d="M 850 300 C 950 300, 950 400, 1050 400" stroke="#8b5cf6" stroke-width="2" fill="none" marker-end="url(#arrowhead)" />
                </svg>

                <div class="grid grid-cols-3 h-full min-w-[1200px] relative z-10 divide-x divide-white/5">
                    <!-- Region 1: Sources -->
                    <div class="bg-black/20 p-8 space-y-6 relative group/region">
                        <div class="absolute top-4 left-4 text-[10px] font-bold uppercase tracking-widest text-foreground/20 group-hover/region:text-foreground/40 transition-colors select-none">
                            REGION: DATA_INGEST
                        </div>
                        
                        <div v-for="sensor in sensors" :key="'canvas-'+sensor.id" 
                            class="w-64 bg-card border border-border shadow-lg rounded-lg p-3 hover:border-primary/50 cursor-grab active:cursor-grabbing hover:-translate-y-1 transition-all duration-300 relative group z-20"
                        >
                             <!-- Port -->
                             <div class="absolute -right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 bg-foreground rounded-full border border-card group-hover:bg-primary transition-colors"></div>
                            
                            <div class="flex items-center gap-3">
                                <div class="p-2 bg-surface-2 rounded-md text-foreground/70">
                                    <component :is="sensor.icon" class="w-4 h-4" />
                                </div>
                                <div>
                                    <div class="text-sm font-bold text-foreground">{{ sensor.name }}</div>
                                    <div class="text-[10px] font-mono text-foreground/50">{{ sensor.type }} / {{ sensor.status }}</div>
                                </div>
                            </div>
                        </div>

                        <div @click="showAddModal = true" class="w-64 border-2 border-dashed border-border/50 rounded-lg p-4 flex items-center justify-center text-foreground/30 hover:text-primary hover:border-primary/30 hover:bg-primary/5 cursor-pointer transition">
                            <Plus class="w-5 h-5" />
                        </div>
                    </div>

                    <!-- Region 2: Processing -->
                    <div class="bg-black/10 p-6 flex flex-col items-center justify-center relative group/region">
                        <div class="absolute top-4 left-4 text-[10px] font-bold uppercase tracking-widest text-foreground/20 group-hover/region:text-foreground/40 transition-colors select-none">
                            REGION: CORE_PROCESSING
                        </div>

                        <div @click="showConfigModal = true" class="w-80 bg-card/90 backdrop-blur border-2 border-primary/20 shadow-2xl rounded-xl p-0 overflow-hidden relative group cursor-grab active:cursor-grabbing hover:border-primary/60 transition-all z-20">
                            <!-- Input Port -->
                            <div class="absolute -left-1.5 top-1/2 -translate-y-1/2 w-3 h-3 bg-primary rounded-full border border-card shadow-[0_0_10px_rgba(var(--primary),0.5)]"></div>
                            <!-- Output Port -->
                            <div class="absolute -right-1.5 top-1/2 -translate-y-1/2 w-3 h-3 bg-primary rounded-full border border-card shadow-[0_0_10px_rgba(var(--primary),0.5)]"></div>

                            <div class="bg-surface-2 p-3 border-b border-border flex justify-between items-center">
                                <span class="font-bold text-sm text-foreground flex items-center gap-2">
                                    <Cpu class="w-4 h-4 text-primary" /> Flowmatic Engine
                                </span>
                                <div class="flex gap-1">
                                    <div class="w-2 h-2 rounded-full bg-green-500 animate-pulse"></div>
                                </div>
                            </div>
                            <div class="p-4 space-y-3">
                                <div class="flex justify-between text-xs">
                                    <span class="text-foreground/50">Anomaly Check</span>
                                    <span class="text-success font-mono">PASS</span>
                                </div>
                                <div class="flex justify-between text-xs">
                                    <span class="text-foreground/50">Schema Clean</span>
                                    <span class="text-success font-mono">PASS</span>
                                </div>
                                <div class="h-px bg-border my-2"></div>
                                <div class="font-mono text-[10px] text-foreground/40">
                                    >> latency: 14ms<br>
                                    >> buffer: 45%
                                </div>
                            </div>
                        </div>
                    </div>

                    <!-- Region 3: Output -->
                    <div class="p-6 space-y-12 flex flex-col justify-center relative group/region bg-black/20">
                         <div class="absolute top-4 left-4 text-[10px] font-bold uppercase tracking-widest text-foreground/20 group-hover/region:text-foreground/40 transition-colors select-none">
                            REGION: DESTINATION
                        </div>

                         <!-- Node 1 -->
                         <div class="w-64 bg-card border border-border shadow-lg rounded-lg p-0 relative ml-12 cursor-grab active:cursor-grabbing hover:-translate-y-1 transition-transform z-20">
                             <div class="absolute -left-1.5 top-1/2 -translate-y-1/2 w-3 h-3 bg-blue-500 rounded-full border border-card"></div>
                            <div class="p-3 flex items-center gap-3">
                                <div class="p-2 bg-blue-500/10 text-blue-500 rounded-md">
                                    <HardDrive class="w-4 h-4" />
                                </div>
                                <div>
                                    <div class="text-sm font-bold text-foreground">S3 Data Lake</div>
                                    <div class="text-[10px] font-mono text-foreground/50">bucket: raw-v2</div>
                                </div>
                            </div>
                             <div class="bg-surface-2 px-3 py-1.5 border-t border-border flex justify-between text-[10px] text-foreground/50 font-mono">
                                <span>Write: 450MB/s</span>
                                <span class="text-blue-400">● Active</span>
                            </div>
                        </div>

                        <!-- Node 2 -->
                        <div class="w-64 bg-card border border-border shadow-lg rounded-lg p-0 relative ml-12 cursor-grab active:cursor-grabbing hover:-translate-y-1 transition-transform z-20">
                            <div class="absolute -left-1.5 top-1/2 -translate-y-1/2 w-3 h-3 bg-violet-500 rounded-full border border-card"></div>
                            <div class="p-3 flex items-center gap-3">
                                <div class="p-2 bg-violet-500/10 text-violet-500 rounded-md">
                                    <Brain class="w-4 h-4" />
                                </div>
                                <div>
                                    <div class="text-sm font-bold text-foreground">Fed. Learning</div>
                                    <div class="text-[10px] font-mono text-foreground/50">cluster: south-1</div>
                                </div>
                            </div>
                            <div class="bg-surface-2 px-3 py-1.5 border-t border-border flex justify-between text-[10px] text-foreground/50 font-mono">
                                <span>Epoch: 42</span>
                                <span class="text-violet-400">● Training</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Main Flow Visualization -->
        <div v-else class="grid lg:grid-cols-12 gap-6 relative">
            
            <!-- Column 1: Sources -->
            <div class="lg:col-span-3 space-y-4 flex flex-col">
                <h3 class="text-xs font-bold text-foreground/50 uppercase tracking-wider mb-2 flex items-center gap-2">
                    <Radio class="w-4 h-4" /> Data Sources ({{ sensors.length }})
                </h3>
                
                <div class="space-y-3 flex-1 overflow-visible">
                    <TransitionGroup name="list">
                        <div v-for="sensor in sensors" :key="sensor.id" 
                            class="bg-card/50 backdrop-blur-sm border border-border p-4 rounded-xl flex items-center gap-4 hover:border-primary/30 transition-all group relative overflow-hidden cursor-pointer"
                        >
                            <!-- Active Indicator -->
                            <div v-if="sensor.status === 'active'" class="absolute left-0 top-0 bottom-0 w-1 bg-success/50" />
                            <div v-else-if="sensor.status === 'warning'" class="absolute left-0 top-0 bottom-0 w-1 bg-warning/50" />
                            <div v-else-if="sensor.status === 'connecting'" class="absolute left-0 top-0 bottom-0 w-1 bg-blue-500/50 animate-pulse" />

                            <div class="p-2.5 rounded-lg bg-surface-2 text-foreground/70 group-hover:text-primary group-hover:bg-primary/10 transition">
                                <component :is="sensor.icon" class="w-5 h-5" />
                            </div>
                            <div class="flex-1 min-w-0">
                                <div class="flex items-center justify-between mb-0.5">
                                    <span class="font-semibold text-sm text-foreground truncate">{{ sensor.name }}</span>
                                    <span v-if="sensor.status === 'active'" class="flex h-2 w-2 rounded-full bg-success shadow-[0_0_8px_rgba(34,197,94,0.6)] animate-pulse"></span>
                                    <span v-else-if="sensor.status === 'connecting'" class="flex h-2 w-2 rounded-full bg-blue-500 shadow-[0_0_8px_rgba(59,130,246,0.6)] animate-spin" style="border-radius:0;"></span>
                                    <span v-else class="flex h-2 w-2 rounded-full bg-warning shadow-[0_0_8px_rgba(234,179,8,0.6)] animate-ping"></span>
                                </div>
                                <div class="flex items-center gap-2 text-xs text-foreground/50 font-mono">
                                    <span>{{ sensor.stats }}</span>
                                    <span class="text-border">|</span>
                                    <span>{{ sensor.status.toUpperCase() }}</span>
                                </div>
                            </div>

                            <!-- Remove Button -->
                            <button @click.stop="removeSensor(sensor.id)" class="absolute right-2 top-2 opacity-0 group-hover:opacity-100 p-1 hover:bg-red-500/20 hover:text-red-500 rounded transition-all">
                                <Trash2 class="w-3.5 h-3.5" />
                            </button>
                        </div>
                    </TransitionGroup>

                    <!-- Add New -->
                    <button @click="showAddModal = true" class="w-full py-3 border border-dashed border-border rounded-xl text-sm text-foreground/50 hover:text-primary hover:border-primary/30 hover:bg-primary/5 transition flex items-center justify-center gap-2 group shadow-sm active:scale-[0.98]">
                         <span class="p-1 rounded-md bg-surface-2 group-hover:bg-primary/20 transition">
                            <Plus class="w-4 h-4" />
                         </span>
                         Connect New Source
                    </button>
                </div>
            </div>

            <!-- Flow connectors (Desktop only) -->
            <div class="hidden lg:flex lg:col-span-1 items-center justify-center relative">
                <div class="w-full h-0.5 bg-gradient-to-r from-border to-primary/50 relative">
                     <div class="absolute inset-0 bg-primary/50 blur-[2px] animate-pulse"></div>
                     <ArrowRight class="absolute right-0 top-1/2 -translate-y-1/2 text-primary w-5 h-5" />
                </div>
            </div>

            <!-- Column 2: Ingestion & Processing -->
            <div class="lg:col-span-4 flex flex-col justify-center space-y-6">
                 <div class="flex items-center justify-between">
                    <h3 class="text-xs font-bold text-foreground/50 uppercase tracking-wider flex items-center gap-2">
                        <Cpu class="w-4 h-4" /> Processing Unit
                    </h3>
                    <div class="flex gap-2">
                         <span class="text-[10px] bg-success/10 text-success px-2 py-0.5 rounded border border-success/20">ONLINE</span>
                    </div>
                 </div>

                <div 
                    @click="showConfigModal = true"
                    class="bg-card/80 backdrop-blur-md border border-primary/20 p-6 rounded-2xl shadow-2xl relative overflow-hidden group cursor-pointer hover:border-primary/50 transition-all active:scale-[0.99]"
                >
                    <div class="absolute inset-x-0 top-0 h-1 bg-gradient-to-r from-primary via-secondary to-primary animate-[shimmer_2s_infinite]"></div>
                    <div class="absolute top-2 right-2 text-primary/30 group-hover:text-primary transition-colors">
                        <Settings class="w-5 h-5" />
                    </div>
                    
                    <div class="flex justify-center mb-6 mt-2">
                        <div class="h-20 w-20 rounded-full bg-primary/10 flex items-center justify-center border border-primary/20 relative transition-transform group-hover:scale-110 duration-500">
                            <Activity class="w-10 h-10 text-primary animate-pulse" />
                            <div class="absolute inset-0 rounded-full border border-primary/30 animate-[ping_3s_infinite]"></div>
                        </div>
                    </div>

                    <div class="space-y-4 text-center relative z-10">
                        <div>
                             <h2 class="text-lg font-bold text-foreground group-hover:text-primary transition-colors">Stream Core Engine</h2>
                             <p class="text-xs text-foreground/50 mt-1">v2.4.1-stable</p>
                        </div>

                        <div class="grid grid-cols-2 gap-2 text-xs">
                            <div class="bg-surface-2 p-3 rounded-lg border border-border group-hover:border-primary/20 transition-colors">
                                <p class="text-foreground/50 mb-1">Throughput</p>
                                <p class="font-mono font-bold text-foreground">{{ streamConfig.throughputLimit * 0.84 }} GB/s</p>
                            </div>
                             <div class="bg-surface-2 p-3 rounded-lg border border-border group-hover:border-primary/20 transition-colors">
                                <p class="text-foreground/50 mb-1">Latency</p>
                                <p class="font-mono font-bold text-foreground">14ms</p>
                            </div>
                        </div>
                        
                        <div class="pt-4 border-t border-border flex flex-col gap-2">
                            <div class="flex items-center justify-between text-sm">
                                <span class="text-foreground/70">Anomaly Detection</span>
                                <span v-if="streamConfig.anomalyDetection" class="text-success font-medium flex items-center gap-1"><Zap class="w-3 h-3" /> Active</span>
                                <span v-else class="text-foreground/40 font-medium">Disabled</span>
                            </div>
                            <div class="flex items-center justify-between text-sm">
                                <span class="text-foreground/70">Schema Validation</span>
                                <span v-if="streamConfig.schemaValidation" class="text-success font-medium">Strict</span>
                                <span v-else class="text-warning font-medium">Lax</span>
                            </div>
                             <div class="flex items-center justify-between text-sm">
                                <span class="text-foreground/70">Refining</span>
                                <span v-if="streamConfig.autoCleaning" class="text-primary font-medium">Auto-Cleaning</span>
                                <span v-else class="text-foreground/40">Manual</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>

            <!-- Flow connectors (Desktop only) -->
            <div class="hidden lg:flex lg:col-span-1 items-center justify-center relative">
                 <!-- Splitter -->
                <div class="relative w-full h-[200px] flex items-center">
                   <svg class="absolute inset-0 w-full h-full overflow-visible" preserveAspectRatio="none">
                        <!-- Top path -->
                        <path d="M0,100 C50,100 50,20 100,20" fill="none" class="stroke-border" stroke-width="2" vector-effect="non-scaling-stroke" />
                         <!-- Bottom path -->
                        <path d="M0,100 C50,100 50,180 100,180" fill="none" class="stroke-border" stroke-width="2" vector-effect="non-scaling-stroke" />
                         
                         <!-- Animated particles -->
                         <circle r="3" fill="var(--color-primary)">
                            <animateMotion dur="2s" repeatCount="indefinite" path="M0,100 C50,100 50,20 100,20" />
                         </circle>
                         <circle r="3" fill="var(--color-secondary)">
                            <animateMotion dur="3s" repeatCount="indefinite" path="M0,100 C50,100 50,180 100,180" />
                         </circle>
                   </svg>
                </div>
            </div>

            <!-- Column 3: Destinations -->
            <div class="lg:col-span-3 flex flex-col justify-between py-4 space-y-6">
                 
                 <!-- Top Branch: Warehouse -->
                 <div class="cursor-pointer group" @click="openDataDestination('Data Lake')">
                    <h3 class="text-xs font-bold text-foreground/50 uppercase tracking-wider mb-2 flex items-center gap-2">
                        <Database class="w-4 h-4" /> Data Lake
                    </h3>
                    <div class="bg-card border border-border p-4 rounded-xl space-y-3 group-hover:border-blue-500/50 transition-all shadow-sm group-hover:shadow-lg group-hover:shadow-blue-500/10">
                        <div class="flex items-center gap-3">
                            <div class="p-2 bg-blue-500/10 rounded-lg text-blue-500">
                                <HardDrive class="w-5 h-5" />
                            </div>
                            <div>
                                <p class="font-semibold text-foreground text-sm">S3 Archive</p>
                                <p class="text-xs text-foreground/50">historical_raw_v2</p>
                            </div>
                        </div>
                         <div class="h-1.5 w-full bg-surface-2 rounded-full overflow-hidden">
                            <div class="h-full bg-blue-500 w-[45%] transition-all duration-1000 group-hover:w-[55%]"></div>
                        </div>
                        <p class="text-xs text-right text-foreground/50">Storage: 45TB / 100TB</p>
                    </div>
                 </div>

                 <!-- Bottom Branch: Federated Learning -->
                 <div class="cursor-pointer group" @click="openDataDestination('Federated Learning')">
                    <h3 class="text-xs font-bold text-foreground/50 uppercase tracking-wider mb-2 flex items-center gap-2">
                        <Network class="w-4 h-4" /> Federated Learning
                    </h3>
                    <div class="bg-gradient-to-br from-violet-500/5 to-fuchsia-500/5 border border-violet-500/20 p-4 rounded-xl space-y-4 group-hover:border-violet-500/50 transition-all shadow-sm group-hover:shadow-lg group-hover:shadow-violet-500/10">
                        <div class="flex items-center gap-3">
                             <div class="p-2 bg-violet-500/10 rounded-lg text-violet-500">
                                <Brain class="w-5 h-5" />
                            </div>
                            <div>
                                <p class="font-semibold text-foreground text-sm">Model Training</p>
                                <p class="text-xs text-foreground/50">Privacy-preserving</p>
                            </div>
                        </div>
                        
                        <div class="flex items-center gap-2 text-xs bg-black/20 p-2 rounded border border-white/5 font-mono text-foreground/70">
                            <ShieldCheck class="w-3 h-3 text-success" />
                            <span>Differential Privacy: Epsilon 2.0</span>
                        </div>

                        <div class="flex items-center justify-between">
                            <div class="flex -space-x-2">
                                <div class="w-6 h-6 rounded-full bg-surface-1 border border-border flex items-center justify-center text-[10px] group-hover:translate-x-1 transition-transform">A</div>
                                <div class="w-6 h-6 rounded-full bg-surface-1 border border-border flex items-center justify-center text-[10px] group-hover:translate-x-1 transition-transform delay-75">B</div>
                                <div class="w-6 h-6 rounded-full bg-surface-1 border border-border flex items-center justify-center text-[10px] group-hover:translate-x-1 transition-transform delay-100">C</div>
                            </div>
                            <div class="flex items-center gap-1 text-xs text-violet-400">
                                <Globe class="w-3 h-3 animate-pulse" />
                                <span>Syncing weights...</span>
                            </div>
                        </div>
                    </div>
                 </div>

            </div>
        </div>

        <!-- Terminal Output -->
        <div class="bg-[#0c0c0c] rounded-xl border border-white/10 p-4 font-mono text-xs text-gray-400 h-64 overflow-hidden relative flex flex-col shadow-2xl">
            <div class="flex justify-between items-center mb-2 pb-2 border-b border-white/5">
                <span class="text-gray-500 uppercase font-bold tracking-wider">System Log</span>
                <div class="flex gap-2">
                    <div class="w-2 h-2 rounded-full bg-red-500/50"></div>
                    <div class="w-2 h-2 rounded-full bg-yellow-500/50"></div>
                    <div class="w-2 h-2 rounded-full bg-green-500/50"></div>
                </div>
            </div>
            <div class="flex-1 overflow-y-auto space-y-1 font-mono">
                <div v-for="(log, i) in logs" :key="i" class="opacity-80 hover:opacity-100 transition-opacity">
                    <span class="text-primary mr-2">➜</span>
                    {{ log }}
                </div>
            </div>
        </div>


        <!-- Overlay Modals -->
        
        <!-- Add Source Modal -->
        <div v-if="showAddModal" class="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div class="absolute inset-0 bg-black/60 backdrop-blur-sm" @click="showAddModal = false"></div>
            <div class="relative w-full max-w-md bg-card border border-border rounded-xl shadow-2xl p-6 space-y-6 animate-in fade-in zoom-in duration-200">
                <div class="flex justify-between items-center">
                    <h2 class="text-xl font-bold text-foreground">Connect New Source</h2>
                    <button @click="showAddModal = false" class="text-foreground/50 hover:text-foreground"><X class="w-5 h-5" /></button>
                </div>
                
                <div class="space-y-4">
                    <div class="space-y-2">
                        <label class="text-sm font-medium text-foreground">Source Name</label>
                        <input v-model="newSourceForm.name" type="text" placeholder="e.g., North Street Cam A" class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground focus:outline-none focus:border-primary/50" />
                    </div>
                    <div class="space-y-2">
                        <label class="text-sm font-medium text-foreground">Source Type</label>
                        <div class="grid grid-cols-2 gap-2">
                            <button 
                                v-for="type in (['video', 'iot', 'power', 'network'] as const)" 
                                :key="type"
                                @click="newSourceForm.type = type"
                                :class="[
                                    'px-3 py-2 rounded-lg border text-sm capitalize transition-all',
                                    newSourceForm.type === type 
                                        ? 'bg-primary/10 border-primary text-primary' 
                                        : 'bg-surface-2 border-border text-foreground/70 hover:border-primary/30'
                                ]"
                            >
                                <component :is="getTypeIcon(type)" class="w-4 h-4 inline mr-1" />
                                {{ type }}
                            </button>
                        </div>
                    </div>
                    <div class="space-y-2">
                         <label class="text-sm font-medium text-foreground">Endpoint URL (Optional)</label>
                         <input v-model="newSourceForm.endpoint" type="text" placeholder="rtsp://..." class="w-full bg-surface-2 border border-border rounded-lg px-3 py-2 text-foreground focus:outline-none focus:border-primary/50" />
                    </div>
                </div>

                <div class="flex justify-end gap-3 pt-2">
                    <button @click="showAddModal = false" class="px-4 py-2 rounded-lg text-foreground/70 hover:bg-surface-2 transition">Cancel</button>
                    <button @click="addSource" class="px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition shadow-lg shadow-primary/20">Connect Source</button>
                </div>
            </div>
        </div>

        <!-- Configure Stream Modal -->
        <div v-if="showConfigModal" class="fixed inset-0 z-50 flex items-center justify-center p-4">
            <div class="absolute inset-0 bg-black/60 backdrop-blur-sm" @click="showConfigModal = false"></div>
            <div class="relative w-full max-w-lg bg-card border border-border rounded-xl shadow-2xl p-6 space-y-6 animate-in fade-in zoom-in duration-200">
                 <div class="flex justify-between items-center bg-surface-2 -mx-6 -mt-6 p-6 border-b border-border rounded-t-xl">
                    <div>
                        <h2 class="text-xl font-bold text-foreground flex items-center gap-2">
                            <Cpu class="w-5 h-5 text-primary" /> Processing Configuration
                        </h2>
                        <p class="text-xs text-foreground/50 mt-1">Adjust real-time engine parameters</p>
                    </div>
                    <button @click="showConfigModal = false" class="text-foreground/50 hover:text-foreground"><X class="w-5 h-5" /></button>
                </div>

                <div class="space-y-6">
                    <!-- Toggles -->
                    <div class="space-y-3">
                         <div class="flex items-center justify-between p-3 rounded-lg border border-border hover:bg-surface-2 transition">
                            <div class="flex items-center gap-3">
                                <div class="p-2 rounded bg-red-500/10 text-red-500"><Activity class="w-4 h-4" /></div>
                                <div>
                                    <p class="font-medium text-foreground text-sm">Anomaly Detection</p>
                                    <p class="text-xs text-foreground/50">Flag outliers in real-time</p>
                                </div>
                            </div>
                            <button @click="streamConfig.anomalyDetection = !streamConfig.anomalyDetection" :class="streamConfig.anomalyDetection ? 'bg-success' : 'bg-surface-3'" class="w-10 h-5 rounded-full relative transition-colors">
                                <div :class="streamConfig.anomalyDetection ? 'translate-x-5' : 'translate-x-1'" class="absolute top-1 left-0 w-3 h-3 bg-white rounded-full transition-transform shadow-sm"></div>
                            </button>
                        </div>

                         <div class="flex items-center justify-between p-3 rounded-lg border border-border hover:bg-surface-2 transition">
                            <div class="flex items-center gap-3">
                                <div class="p-2 rounded bg-blue-500/10 text-blue-500"><ShieldCheck class="w-4 h-4" /></div>
                                <div>
                                    <p class="font-medium text-foreground text-sm">Schema Validation</p>
                                    <p class="text-xs text-foreground/50">Enforce strict typing</p>
                                </div>
                            </div>
                            <button @click="streamConfig.schemaValidation = !streamConfig.schemaValidation" :class="streamConfig.schemaValidation ? 'bg-success' : 'bg-surface-3'" class="w-10 h-5 rounded-full relative transition-colors">
                                <div :class="streamConfig.schemaValidation ? 'translate-x-5' : 'translate-x-1'" class="absolute top-1 left-0 w-3 h-3 bg-white rounded-full transition-transform shadow-sm"></div>
                            </button>
                        </div>
                    </div>

                    <!-- Slider -->
                    <div class="space-y-3 pt-2">
                        <div class="flex justify-between text-sm">
                            <span class="text-foreground font-medium">Throughput Limit</span>
                            <span class="font-mono text-primary">{{ streamConfig.throughputLimit }} GB/s</span>
                        </div>
                        <input 
                            v-model.number="streamConfig.throughputLimit" 
                            type="range" min="1" max="20" step="0.5" 
                            class="w-full h-2 bg-surface-3 rounded-lg appearance-none cursor-pointer accent-primary" 
                        />
                         <div class="flex justify-between text-xs text-foreground/40 font-mono">
                            <span>1 GB/s</span>
                            <span>20 GB/s</span>
                        </div>
                    </div>
                </div>

                <div class="flex justify-end gap-3 pt-4 border-t border-border">
                    <button @click="showConfigModal = false" class="px-4 py-2 rounded-lg text-foreground/70 hover:bg-surface-2 transition">Cancel</button>
                    <button @click="saveConfiguration" class="px-4 py-2 rounded-lg bg-primary text-primary-foreground font-medium hover:bg-primary/90 transition shadow-lg shadow-primary/20 flex items-center gap-2">
                        <Save class="w-4 h-4" /> Apply Changes
                    </button>
                </div>
            </div>
        </div>

    </div>
</template>

<style scoped>
@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

/* List Transitions */
.list-move,
.list-enter-active,
.list-leave-active {
  transition: all 0.5s ease;
}

.list-enter-from,
.list-leave-to {
  opacity: 0;
  transform: translateX(-30px);
}

.list-leave-active {
  position: absolute;
}
</style>