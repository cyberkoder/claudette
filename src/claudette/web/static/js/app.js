/**
 * Claudette Dashboard - Alpine.js Application
 */

function claudetteDashboard() {
    return {
        // Current page
        currentPage: 'status',

        // WebSocket connection
        ws: null,
        wsConnected: false,
        wsReconnectAttempts: 0,
        wsMaxReconnectAttempts: 10,
        wsReconnectDelay: 1000,

        // State from server
        state: {
            state: 'idle',
            conversation_mode: false,
            awaiting_confirmation: false,
            audio_level: 0,
            last_transcription: '',
            last_response: '',
            uptime_seconds: 0,
            timestamp: ''
        },

        // Local uptime tracking
        _uptimeBase: 0,
        _uptimeTimestamp: null,
        _uptimeInterval: null,

        // Audio level smoothing
        _targetAudioLevel: 0,
        _audioDecayInterval: null,

        // Configuration
        config: {
            whisper: {},
            vad: {},
            tts: {},
            wake_word: {},
            memory: {},
            sounds: {},
            hotkey: {}
        },

        // Claude activity tracking
        claudeActivity: {
            active: false,
            query: '',
            status: '',
            current_output: '',
            progress_lines: [],
            started_at: '',
            elapsed_seconds: 0
        },

        // Data
        voices: [],
        personalities: [],
        skillsData: [],
        mcpTools: [],
        history: [],
        logs: [],
        systemInfo: null,

        // Skills page state
        skillSearch: '',
        skillFilter: 'all',
        expandedSkill: null,

        // Skill categories
        skillCategories: [
            { id: 'info', name: 'Information', icon: 'i', color: 'var(--accent-blue)' },
            { id: 'system', name: 'System Control', icon: 'S', color: 'var(--accent-green)' },
            { id: 'media', name: 'Media', icon: 'M', color: 'var(--accent-purple)' },
            { id: 'social', name: 'Social', icon: 'C', color: 'var(--accent-pink)' },
            { id: 'memory', name: 'Memory', icon: 'm', color: 'var(--accent-yellow)' },
            { id: 'settings', name: 'Settings', icon: 'G', color: '#6e7681' },
            { id: 'help', name: 'Help', icon: '?', color: 'var(--text-muted)' },
            { id: 'builtin', name: 'Other', icon: 'O', color: 'var(--border-color)' },
        ],

        // UI state
        autoScrollLogs: true,
        wakeWordInput: '',
        selectedVoice: 'en-GB-SoniaNeural',
        selectedRate: '+0%',
        testingVoice: false,
        testAudio: null,

        // Computed properties
        get stateClass() {
            const stateMap = {
                'Listening for': 'listening',
                'Listening (conversation': 'listening',
                'Recording': 'recording',
                'Transcribing': 'processing',
                'Thinking': 'processing',
                'Speaking': 'speaking'
            };

            for (const [key, value] of Object.entries(stateMap)) {
                if (this.state.state && this.state.state.includes(key)) {
                    return value;
                }
            }
            return 'idle';
        },

        get stateIcon() {
            const iconMap = {
                'listening': '\u{1F3A4}',  // microphone
                'recording': '\u{1F534}',  // red circle
                'processing': '\u{23F3}',  // hourglass
                'speaking': '\u{1F5E3}',   // speaking head
                'idle': '\u{1F4A4}'        // zzz
            };
            return iconMap[this.stateClass] || '\u{2753}';
        },

        // Filtered skills based on search
        get filteredSkills() {
            if (!this.skillSearch) return this.skillsData;
            const search = this.skillSearch.toLowerCase();
            return this.skillsData.filter(s =>
                s.name.toLowerCase().includes(search) ||
                s.description.toLowerCase().includes(search) ||
                s.triggers.some(t => t.toLowerCase().includes(search))
            );
        },

        // Filtered MCP tools based on search
        get filteredMCPTools() {
            if (!this.skillSearch) return this.mcpTools;
            const search = this.skillSearch.toLowerCase();
            return this.mcpTools.filter(t =>
                t.name.toLowerCase().includes(search) ||
                t.description.toLowerCase().includes(search) ||
                t.examples.some(e => e.toLowerCase().includes(search))
            );
        },

        // Initialization
        async init() {
            this.connectWebSocket();
            await this.loadInitialData();

            // Set up periodic system info refresh
            setInterval(() => this.loadSystemInfo(), 30000);

            // Start local uptime ticker (every second)
            this._uptimeInterval = setInterval(() => {
                if (this._uptimeTimestamp) {
                    const elapsed = (Date.now() - this._uptimeTimestamp) / 1000;
                    this.state.uptime_seconds = this._uptimeBase + elapsed;
                }
            }, 1000);

            // Audio level smoothing - interpolate towards target, decay when idle
            this._audioDecayInterval = setInterval(() => {
                const current = this.state.audio_level;
                const target = this._targetAudioLevel;

                if (Math.abs(current - target) > 0.01) {
                    // Smoothly move towards target
                    this.state.audio_level = current + (target - current) * 0.3;
                } else if (target < 0.05) {
                    // Decay slowly when quiet
                    this.state.audio_level = Math.max(0, current * 0.85);
                }
            }, 50); // 20fps for smooth animation
        },

        // WebSocket connection
        connectWebSocket() {
            const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
            const wsUrl = `${protocol}//${window.location.host}/ws`;

            try {
                this.ws = new WebSocket(wsUrl);

                this.ws.onopen = () => {
                    console.log('WebSocket connected');
                    this.wsConnected = true;
                    this.wsReconnectAttempts = 0;
                };

                this.ws.onclose = () => {
                    console.log('WebSocket disconnected');
                    this.wsConnected = false;
                    this.scheduleReconnect();
                };

                this.ws.onerror = (error) => {
                    console.error('WebSocket error:', error);
                };

                this.ws.onmessage = (event) => {
                    this.handleWebSocketMessage(JSON.parse(event.data));
                };
            } catch (error) {
                console.error('Failed to connect WebSocket:', error);
                this.scheduleReconnect();
            }
        },

        scheduleReconnect() {
            if (this.wsReconnectAttempts >= this.wsMaxReconnectAttempts) {
                console.log('Max reconnect attempts reached');
                return;
            }

            this.wsReconnectAttempts++;
            const delay = this.wsReconnectDelay * Math.pow(2, this.wsReconnectAttempts - 1);
            console.log(`Reconnecting in ${delay}ms (attempt ${this.wsReconnectAttempts})`);

            setTimeout(() => this.connectWebSocket(), delay);
        },

        handleWebSocketMessage(message) {
            switch (message.type) {
                case 'connected':
                    if (message.data.initial_state) {
                        Object.assign(this.state, message.data.initial_state);
                        // Store base uptime for local tracking
                        this._uptimeBase = message.data.initial_state.uptime_seconds || 0;
                        this._uptimeTimestamp = Date.now();
                    }
                    if (message.data.claude_activity) {
                        Object.assign(this.claudeActivity, message.data.claude_activity);
                    }
                    break;

                case 'state':
                    // Update uptime base when we get server updates
                    if (message.data.uptime_seconds !== undefined) {
                        this._uptimeBase = message.data.uptime_seconds;
                        this._uptimeTimestamp = Date.now();
                    }
                    Object.assign(this.state, message.data);
                    break;

                case 'audio_level':
                    this._targetAudioLevel = message.data.level;
                    break;

                case 'log':
                    this.logs.push(message.data);
                    // Keep only last 200 logs in memory
                    if (this.logs.length > 200) {
                        this.logs = this.logs.slice(-200);
                    }
                    // Auto-scroll if enabled
                    if (this.autoScrollLogs && this.$refs.logsContainer) {
                        this.$nextTick(() => {
                            this.$refs.logsContainer.scrollTop = this.$refs.logsContainer.scrollHeight;
                        });
                    }
                    break;

                case 'claude_activity':
                    Object.assign(this.claudeActivity, message.data);
                    // Auto-scroll the output
                    if (this.$refs.claudeOutput) {
                        this.$nextTick(() => {
                            this.$refs.claudeOutput.scrollTop = this.$refs.claudeOutput.scrollHeight;
                        });
                    }
                    break;

                case 'pong':
                    // Heartbeat response
                    break;

                default:
                    console.log('Unknown message type:', message.type);
            }
        },

        // Data loading
        async loadInitialData() {
            await Promise.all([
                this.loadConfig(),
                this.loadVoices(),
                this.loadPersonalities(),
                this.loadSkills(),
                this.loadHistory(),
                this.loadSystemInfo(),
                this.loadLogs(),
                this.loadClaudeActivity()
            ]);
        },

        async loadConfig() {
            try {
                const response = await fetch('/api/config');
                if (response.ok) {
                    this.config = await response.json();
                    // Initialize voice settings from config
                    if (this.config.tts?.voice) {
                        this.selectedVoice = this.config.tts.voice;
                    }
                    if (this.config.tts?.rate) {
                        this.selectedRate = this.config.tts.rate;
                    }
                }
            } catch (error) {
                console.error('Failed to load config:', error);
            }
        },

        async loadVoices() {
            try {
                const response = await fetch('/api/voices');
                if (response.ok) {
                    this.voices = await response.json();
                }
            } catch (error) {
                console.error('Failed to load voices:', error);
            }
        },

        async loadPersonalities() {
            try {
                const response = await fetch('/api/personalities');
                if (response.ok) {
                    this.personalities = await response.json();
                }
            } catch (error) {
                console.error('Failed to load personalities:', error);
            }
        },

        async loadSkills() {
            try {
                const response = await fetch('/api/skills');
                if (response.ok) {
                    const data = await response.json();
                    this.skillsData = data.skills || [];
                    this.mcpTools = data.mcp_tools || [];
                }
            } catch (error) {
                console.error('Failed to load skills:', error);
            }
        },

        // Get skills filtered by category
        getSkillsByCategory(categoryId) {
            return this.filteredSkills.filter(s => s.category === categoryId);
        },

        // Format skill name for display (snake_case to Title Case)
        formatSkillName(name) {
            return name.split('_')
                .map(word => word.charAt(0).toUpperCase() + word.slice(1))
                .join(' ');
        },

        async loadHistory() {
            try {
                const response = await fetch('/api/history?limit=50');
                if (response.ok) {
                    const data = await response.json();
                    this.history = data.entries;
                }
            } catch (error) {
                console.error('Failed to load history:', error);
            }
        },

        async loadSystemInfo() {
            try {
                const response = await fetch('/api/system');
                if (response.ok) {
                    this.systemInfo = await response.json();
                }
            } catch (error) {
                console.error('Failed to load system info:', error);
            }
        },

        async loadLogs() {
            try {
                const response = await fetch('/api/logs?limit=100');
                if (response.ok) {
                    const data = await response.json();
                    this.logs = data.entries;
                }
            } catch (error) {
                console.error('Failed to load logs:', error);
            }
        },

        async loadClaudeActivity() {
            try {
                const response = await fetch('/api/claude/activity');
                if (response.ok) {
                    const data = await response.json();
                    Object.assign(this.claudeActivity, data);
                }
            } catch (error) {
                console.error('Failed to load Claude activity:', error);
            }
        },

        // Config updates
        async updateConfig(key, value) {
            try {
                const response = await fetch('/api/config', {
                    method: 'PATCH',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ updates: { [key]: value } })
                });

                if (response.ok) {
                    const data = await response.json();
                    this.config = data.config;
                    console.log(`Updated ${key} to ${value}`);
                } else {
                    console.error('Failed to update config');
                }
            } catch (error) {
                console.error('Failed to update config:', error);
            }
        },

        // Actions
        async testVoice() {
            if (this.testingVoice) return;

            this.testingVoice = true;

            try {
                const voice = this.selectedVoice || this.config.tts?.voice || 'en-GB-SoniaNeural';
                const rate = this.selectedRate || this.config.tts?.rate || '+0%';

                const url = `/api/voice/test?voice=${encodeURIComponent(voice)}&rate=${encodeURIComponent(rate)}`;
                const response = await fetch(url);

                if (!response.ok) {
                    throw new Error('Failed to generate audio');
                }

                const blob = await response.blob();
                const audioUrl = URL.createObjectURL(blob);

                // Stop any existing audio
                if (this.testAudio) {
                    this.testAudio.pause();
                    URL.revokeObjectURL(this.testAudio.src);
                }

                this.testAudio = new Audio(audioUrl);
                this.testAudio.onended = () => {
                    this.testingVoice = false;
                    URL.revokeObjectURL(audioUrl);
                };
                this.testAudio.onerror = () => {
                    this.testingVoice = false;
                    console.error('Audio playback error');
                };

                await this.testAudio.play();

            } catch (error) {
                console.error('Voice test error:', error);
                this.testingVoice = false;
            }
        },

        async updateWakeWord() {
            const newWord = this.wakeWordInput.trim().toLowerCase();
            if (!newWord) {
                return;
            }

            await this.updateConfig('wake_word.word', newWord);
            this.wakeWordInput = '';
        },

        async clearHistory() {
            if (!confirm('Are you sure you want to clear all conversation history?')) {
                return;
            }

            try {
                const response = await fetch('/api/history/clear', { method: 'POST' });
                if (response.ok) {
                    this.history = [];
                    await this.loadConfig(); // Refresh memory count
                }
            } catch (error) {
                console.error('Failed to clear history:', error);
            }
        },

        // Formatting helpers
        formatUptime(seconds) {
            if (!seconds) return '0:00:00';

            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            const secs = Math.floor(seconds % 60);

            return `${hours}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
        },

        formatDate(isoString) {
            if (!isoString) return '';
            const date = new Date(isoString);
            return date.toLocaleString();
        },

        formatLogTime(isoString) {
            if (!isoString) return '';
            const date = new Date(isoString);
            return date.toLocaleTimeString();
        },

        formatElapsed(seconds) {
            if (!seconds || seconds < 0) return '0s';
            if (seconds < 60) {
                return Math.floor(seconds) + 's';
            }
            const mins = Math.floor(seconds / 60);
            const secs = Math.floor(seconds % 60);
            return `${mins}m ${secs}s`;
        }
    };
}
