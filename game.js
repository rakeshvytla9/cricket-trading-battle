// ===== CRICKET TRADING BATTLE - ML-POWERED SIMULATION =====
// Features:
// 1. Geometric Brownian Motion (GBM) for realistic price movements
// 2. **ML Model** trained on 67,303 IPL deliveries (2020-2024)
// 3. Momentum indicators (EMA) for price drift
// 4. Mean reversion toward base prices
// 5. Bowler-Batsman matchup data from real historical encounters

// ===== ML PREDICTOR INITIALIZATION =====
let mlPredictor = null;
let mlPredictorReady = false;

// Initialize ML Predictor (async)
const initMLPredictor = async () => {
    if (typeof CricketMLPredictor !== 'undefined') {
        mlPredictor = new CricketMLPredictor();
        mlPredictorReady = await mlPredictor.load();
        if (mlPredictorReady) {
            console.log('🤖 ML Predictor loaded - using trained model!');
        }
    } else {
        console.log('📊 Using fallback statistical model');
    }
};

// Start loading ML predictor immediately
// Start loading ML predictor immediately
initMLPredictor();

// ===== GEMINI 3 INTEGRATION =====
// API Key is now handled securely on the server (Vercel)
let gemini = null;

if (typeof GeminiAI !== 'undefined') {
    gemini = new GeminiAI(); // No key needed on client
    console.log('✨ Gemini 3 AI Initialized (Server-Side Mode)');
}

// Chat UI Elements
const chatElements = {
    panel: document.getElementById('ai-chat'),
    toggle: document.getElementById('chat-toggle'),
    body: document.getElementById('chat-body'),
    input: document.getElementById('chat-input'),
    send: document.getElementById('chat-send')
};

// Toggle Chat
if (chatElements.toggle) {
    chatElements.toggle.addEventListener('click', () => {
        chatElements.panel.classList.toggle('minimized');
        chatElements.toggle.textContent = chatElements.panel.classList.contains('minimized') ? '+' : '−';
    });
}

// Send Message
const sendChatMessage = async () => {
    const text = chatElements.input.value.trim();
    if (!text || !gemini) return;

    // User Message
    appendChatMessage('user', text);
    chatElements.input.value = '';

    // Loading state
    const loadingId = appendChatMessage('ai', 'Thinking...');

    // Gemini Call
    const response = await gemini.getTradingAdvice(text, gameState, {
        stocks: gameState.yourPortfolio,
        holdings: gameState.yourPortfolio, // simplified
        cash: gameState.yourCash
    });

    // Update AI Message
    const msgDiv = document.querySelector(`[data-msg-id="${loadingId}"] .message-content`);
    if (msgDiv) msgDiv.textContent = response;
};

const appendChatMessage = (role, text) => {
    const div = document.createElement('div');
    div.className = `chat-message ${role}`;
    const id = Date.now();
    div.dataset.msgId = id;
    div.innerHTML = `<div class="message-content">${text}</div>`;
    chatElements.body.appendChild(div);
    chatElements.body.scrollTop = chatElements.body.scrollHeight;
    return id;
};

if (chatElements.send) {
    chatElements.send.addEventListener('click', sendChatMessage);
    chatElements.input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage();
    });
}

// ===== IPL PLAYER STATISTICS (Real Data) =====
// ===== FULL IPL SQUADS (RCB vs CSK) =====
const FULL_TEAMS = {
    RCB: [
        { name: 'V. Kohli', role: 'batsman', price: 100, stats: { dotPct: 0.38, singlePct: 0.32, fourPct: 0.12, sixPct: 0.045, wicketPct: 0.027, volatility: 0.15 } },
        { name: 'F. du Plessis', role: 'batsman', price: 95, stats: { dotPct: 0.35, singlePct: 0.35, fourPct: 0.13, sixPct: 0.05, wicketPct: 0.03, volatility: 0.16 } },
        { name: 'R. Patidar', role: 'batsman', price: 80, stats: { dotPct: 0.40, singlePct: 0.25, fourPct: 0.10, sixPct: 0.07, wicketPct: 0.04, volatility: 0.18 } },
        { name: 'G. Maxwell', role: 'allrounder', price: 110, stats: { dotPct: 0.45, singlePct: 0.15, fourPct: 0.15, sixPct: 0.09, wicketPct: 0.06, volatility: 0.25 } },
        { name: 'C. Green', role: 'allrounder', price: 90, stats: { dotPct: 0.35, singlePct: 0.30, fourPct: 0.11, sixPct: 0.05, wicketPct: 0.035, volatility: 0.20 } },
        { name: 'D. Karthik', role: 'batsman', price: 85, stats: { dotPct: 0.30, singlePct: 0.25, fourPct: 0.16, sixPct: 0.08, wicketPct: 0.05, volatility: 0.22 } },
        { name: 'M. Lomror', role: 'batsman', price: 50, stats: { dotPct: 0.35, singlePct: 0.35, fourPct: 0.08, sixPct: 0.04, wicketPct: 0.045, volatility: 0.30 } },
        { name: 'K. Sharma', role: 'bowler', price: 45, stats: { economy: 8.5, wicketPct: 0.05, volatility: 0.25 } },
        { name: 'M. Siraj', role: 'bowler', price: 85, stats: { economy: 7.5, wicketPct: 0.055, volatility: 0.15 } },
        { name: 'L. Ferguson', role: 'bowler', price: 75, stats: { economy: 8.8, wicketPct: 0.06, volatility: 0.20 } },
        { name: 'Y. Dayal', role: 'bowler', price: 60, stats: { economy: 8.0, wicketPct: 0.05, volatility: 0.18 } }
    ],
    CSK: [
        { name: 'R. Gaikwad', role: 'batsman', price: 95, stats: { dotPct: 0.36, singlePct: 0.38, fourPct: 0.13, sixPct: 0.03, wicketPct: 0.025, volatility: 0.14 } },
        { name: 'R. Ravindra', role: 'batsman', price: 85, stats: { dotPct: 0.40, singlePct: 0.25, fourPct: 0.14, sixPct: 0.06, wicketPct: 0.04, volatility: 0.19 } },
        { name: 'A. Rahane', role: 'batsman', price: 70, stats: { dotPct: 0.38, singlePct: 0.30, fourPct: 0.12, sixPct: 0.04, wicketPct: 0.035, volatility: 0.16 } },
        { name: 'S. Dube', role: 'batsman', price: 90, stats: { dotPct: 0.45, singlePct: 0.15, fourPct: 0.10, sixPct: 0.12, wicketPct: 0.05, volatility: 0.24 } },
        { name: 'D. Mitchell', role: 'allrounder', price: 85, stats: { dotPct: 0.35, singlePct: 0.35, fourPct: 0.10, sixPct: 0.04, wicketPct: 0.03, volatility: 0.17 } },
        { name: 'R. Jadeja', role: 'allrounder', price: 100, stats: { economy: 7.0, dotPct: 0.30, singlePct: 0.40, fourPct: 0.08, sixPct: 0.04, wicketPct: 0.03, volatility: 0.12 } },
        { name: 'MS Dhoni', role: 'batsman', price: 120, stats: { dotPct: 0.50, singlePct: 0.10, fourPct: 0.15, sixPct: 0.10, wicketPct: 0.08, volatility: 0.20 } },
        { name: 'S. Thakur', role: 'bowler', price: 75, stats: { economy: 9.0, wicketPct: 0.065, volatility: 0.22 } },
        { name: 'D. Chahar', role: 'bowler', price: 80, stats: { economy: 7.8, wicketPct: 0.058, volatility: 0.16 } },
        { name: 'T. Deshpande', role: 'bowler', price: 65, stats: { economy: 8.5, wicketPct: 0.052, volatility: 0.20 } },
        { name: 'M. Pathirana', role: 'bowler', price: 95, stats: { economy: 7.2, wicketPct: 0.07, volatility: 0.18 } }
    ]
};

// ===== MATCH CONTEXT MODIFIERS =====
const MATCH_CONTEXT = {
    powerplay: { // Overs 1-6
        fourMod: 1.15,
        sixMod: 1.10,
        wicketMod: 1.20,
        dotMod: 0.90
    },
    middle: { // Overs 7-15
        fourMod: 0.95,
        sixMod: 0.90,
        wicketMod: 0.85,
        dotMod: 1.10
    },
    death: { // Overs 16-20
        fourMod: 1.20,
        sixMod: 1.40,
        wicketMod: 1.15,
        dotMod: 0.75
    }
};

// ===== GAME STATE =====
const gameState = {
    // Match State
    innings: 1,
    battingTeam: 'RCB',
    bowlingTeam: 'CSK',
    runs: 0,
    wickets: 0,
    balls: 0,
    overs: 0,
    target: null,
    firstInningsScore: 0,

    // Timer
    ballTimer: 5,
    timerInterval: null,

    // Trading State
    yourCash: 10000,
    yourPortfolio: {},
    yourProfit: 0,
    yourRealizedProfit: 0,

    cpuCash: 10000,
    cpuPortfolio: {},
    cpuProfit: 0,
    cpuRealizedProfit: 0,

    // Players & Indices
    players: [],
    strikerIndex: 0,     // Index in players array
    nonStrikerIndex: 1,  // Index in players array
    currentBowlerIndex: 0, // Relative index in bowling team

    // Game settings
    isRunning: false,

    // Stats
    totalTrades: 0,
    cpuTrades: 0,
    sixCount: 0,
    fourCount: 0,

    // Selected player for trading
    selectedPlayer: null
};

// ===== CREATE PLAYERS =====
const createPlayers = () => {
    const players = [];

    // Helper to add a team
    const addTeam = (teamName, squad) => {
        squad.forEach((p, idx) => {
            // Default batting stats if missing
            if (!p.stats.dotPct && p.role === 'bowler') {
                p.stats = { ...p.stats, dotPct: 0.6, singlePct: 0.2, fourPct: 0.05, sixPct: 0.01, wicketPct: 0.1 };
            }
            // Default bowling stats if missing
            if (!p.stats.economy && p.role === 'batsman') {
                p.stats = { ...p.stats, economy: 9.0, wicketPct: 0.02 };
            }

            players.push({
                id: `${teamName}_${idx}`,
                name: p.name,
                team: teamName,
                role: p.role,
                stats: p.stats,
                basePrice: p.price,
                price: p.price,
                priceHistory: [p.price],
                momentum: 0,
                runs: 0,
                balls: 0,
                wickets: 0,
                runsConceded: 0,
                overs: 0,
                isOut: false,
                volatility: p.stats.volatility || 0.15
            });
        });
    };

    addTeam('RCB', FULL_TEAMS.RCB);
    addTeam('CSK', FULL_TEAMS.CSK);

    return players;
};

// ===== MATHEMATICAL UTILITIES =====

// Box-Muller transform for normal distribution
const normalRandom = () => {
    let u = 0, v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
};

// Exponential Moving Average
const calculateEMA = (data, period = 3) => {
    if (data.length === 0) return 0;
    if (data.length === 1) return data[0];

    const multiplier = 2 / (period + 1);
    let ema = data[0];

    for (let i = 1; i < data.length; i++) {
        ema = (data[i] - ema) * multiplier + ema;
    }

    return ema;
};

// Calculate momentum from price history
const calculateMomentum = (player) => {
    const history = player.priceHistory;
    if (history.length < 3) return 0;

    const recentPrices = history.slice(-5);
    const ema = calculateEMA(recentPrices, 3);
    const currentPrice = history[history.length - 1];

    // Momentum is the percentage difference from EMA
    return (currentPrice - ema) / ema;
};

// ===== ML-POWERED BALL OUTCOME =====
const getDataDrivenOutcome = (batsman, bowler, balls) => {
    const batsmanStats = batsman.stats;
    const bowlerStats = bowler.stats;
    const over = Math.floor(balls / 6);
    const ballInOver = (balls % 6) + 1;

    // Try ML predictor first if available
    if (mlPredictorReady && mlPredictor) {
        try {
            const matchState = {
                over: over,
                ball: ballInOver,
                innings: gameState.innings,
                runsScored: gameState.runs,
                wicketsLost: gameState.wickets,
                target: gameState.target || 170,
                batsmanName: batsman.name,
                bowlerName: bowler.name,
                venueAvgScore: 165,
                recentForm: batsman.runs > 0 ? Math.min(batsman.runs / batsman.balls * 100, 200) : 0
            };

            const prediction = mlPredictor.predict(matchState);

            // Convert ML outcome to game format
            const outcomeMap = {
                'DOT': 0,
                'SINGLE': 1,
                'DOUBLE': 2,
                'TRIPLE': 3,
                'FOUR': 4,
                'SIX': 6,
                'WICKET': 'W'
            };

            return outcomeMap[prediction.outcome];
        } catch (e) {
            console.warn('ML prediction failed, using fallback:', e);
        }
    }

    // Fallback: Original statistical model
    let context;
    if (over < 6) {
        context = MATCH_CONTEXT.powerplay;
    } else if (over < 15) {
        context = MATCH_CONTEXT.middle;
    } else {
        context = MATCH_CONTEXT.death;
    }

    const combinedProbs = {
        dot: (batsmanStats.dotPct * 0.4 + bowlerStats.dotPct * 0.6) * context.dotMod,
        single: (batsmanStats.singlePct + bowlerStats.singlePct) / 2,
        two: (batsmanStats.twoPct + bowlerStats.twoPct) / 2,
        three: (batsmanStats.threePct + bowlerStats.threePct) / 2,
        four: (batsmanStats.fourPct * 0.6 + bowlerStats.fourPct * 0.4) * context.fourMod,
        six: (batsmanStats.sixPct * 0.7 + bowlerStats.sixPct * 0.3) * context.sixMod,
        wicket: (batsmanStats.wicketPct * 0.5 + bowlerStats.wicketPct * 0.5) * context.wicketMod
    };

    const total = Object.values(combinedProbs).reduce((a, b) => a + b, 0);
    Object.keys(combinedProbs).forEach(key => {
        combinedProbs[key] /= total;
    });

    const rand = Math.random();
    let cumulative = 0;

    cumulative += combinedProbs.dot;
    if (rand < cumulative) return 0;

    cumulative += combinedProbs.single;
    if (rand < cumulative) return 1;

    cumulative += combinedProbs.two;
    if (rand < cumulative) return 2;

    cumulative += combinedProbs.three;
    if (rand < cumulative) return 3;

    cumulative += combinedProbs.four;
    if (rand < cumulative) return 4;

    cumulative += combinedProbs.six;
    if (rand < cumulative) return 6;

    return 'W';
};

// ===== GEOMETRIC BROWNIAN MOTION PRICE ENGINE =====
const updatePriceGBM = (player, event, allPlayers) => {
    const dt = 1 / 120; // Time step (1 ball out of 120 in a T20 match)
    const sigma = player.volatility;

    // Calculate drift from momentum
    const momentum = calculateMomentum(player);
    const baseDrift = momentum * 0.5; // Momentum contributes to drift

    // Event shock (immediate price impact)
    let eventShock = 0;
    if (player.role === 'batsman') {
        switch (event) {
            case 6: eventShock = 0.08 + Math.random() * 0.04; break;
            case 4: eventShock = 0.04 + Math.random() * 0.02; break;
            case 'W': eventShock = -0.15 - Math.random() * 0.10; break;
            case 0: eventShock = -0.01 - Math.random() * 0.01; break;
            case 1: case 2: case 3: eventShock = 0.01 + Math.random() * 0.01; break;
        }
    } else { // Bowler
        switch (event) {
            case 6: eventShock = -0.06 - Math.random() * 0.03; break;
            case 4: eventShock = -0.03 - Math.random() * 0.02; break;
            case 'W': eventShock = 0.10 + Math.random() * 0.08; break;
            case 0: eventShock = 0.02 + Math.random() * 0.01; break;
            case 1: case 2: case 3: eventShock = -0.005 - Math.random() * 0.005; break;
        }
    }

    // Mean reversion component (prices tend to return to base)
    const meanReversionStrength = 0.02;
    const priceRatio = player.price / player.basePrice;
    const meanReversion = -meanReversionStrength * Math.log(priceRatio);

    // Market correlation (all players slightly affected by market sentiment)
    const marketSentiment = (Math.random() - 0.5) * 0.01;

    // Combined drift
    const mu = baseDrift + meanReversion + marketSentiment;

    // GBM formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z + shock)
    const Z = normalRandom();
    const randomComponent = sigma * Math.sqrt(dt) * Z;
    const deterministicComponent = (mu - 0.5 * sigma * sigma) * dt;

    const newPrice = player.price * Math.exp(deterministicComponent + randomComponent + eventShock);

    // Clamp price between 10 and 500
    return Math.max(10, Math.min(500, Math.round(newPrice)));
};

// ===== UPDATE ALL PRICES =====
const updatePrices = (outcome, batsmanIndex, bowlerIndex) => {
    const batsman = gameState.players[batsmanIndex];
    const bowler = gameState.players[bowlerIndex];

    // Update involved players with GBM
    const oldBatsmanPrice = batsman.price;
    const oldBowlerPrice = bowler.price;

    batsman.price = updatePriceGBM(batsman, outcome, gameState.players);
    bowler.price = updatePriceGBM(bowler, outcome, gameState.players);

    batsman.priceHistory.push(batsman.price);
    bowler.priceHistory.push(bowler.price);

    // Keep history limited
    if (batsman.priceHistory.length > 30) batsman.priceHistory.shift();
    if (bowler.priceHistory.length > 30) bowler.priceHistory.shift();

    // Update other players with small GBM movements (market noise)
    gameState.players.forEach((player, idx) => {
        if (idx !== batsmanIndex && idx !== bowlerIndex) {
            player.price = updatePriceGBM(player, null, gameState.players);
            player.priceHistory.push(player.price);
            if (player.priceHistory.length > 30) player.priceHistory.shift();
        }
    });

    return {
        batsmanPriceChange: batsman.price - oldBatsmanPrice,
        bowlerPriceChange: bowler.price - oldBowlerPrice
    };
};

// ===== DOM ELEMENTS =====
const elements = {
    startScreen: document.getElementById('start-screen'),
    gameScreen: document.getElementById('game-screen'),
    endScreen: document.getElementById('end-screen'),
    startBtn: document.getElementById('start-btn'),
    playAgainBtn: document.getElementById('play-again-btn'),

    matchStage: document.getElementById('match-stage'),
    score: document.getElementById('score'),
    overs: document.getElementById('overs'),
    runRate: document.getElementById('run-rate'),
    requiredRate: document.getElementById('required-rate'),
    timer: document.getElementById('timer'),

    yourProfit: document.getElementById('your-profit'),
    cpuProfit: document.getElementById('cpu-profit'),
    indicatorFill: document.getElementById('indicator-fill'),

    strikerName: document.getElementById('striker-name'),
    strikerScore: document.getElementById('striker-score'),
    bowlerName: document.getElementById('bowler-name'),
    bowlerFigures: document.getElementById('bowler-figures'),

    ballResult: document.getElementById('ball-result'),
    commentary: document.getElementById('commentary'),
    stocksGrid: document.getElementById('stocks-grid'),
    yourCash: document.getElementById('your-cash'),
    portfolioList: document.getElementById('portfolio-list'),
    totalInvested: document.getElementById('total-invested'),
    currentValue: document.getElementById('current-value'),
    unrealizedPnl: document.getElementById('unrealized-pnl'),

    modal: document.getElementById('trade-modal'),
    modalPlayerName: document.getElementById('modal-player-name'),
    modalPrice: document.getElementById('modal-price'),
    modalChange: document.getElementById('modal-change'),
    modalHoldings: document.getElementById('modal-holdings'),
    tradeQty: document.getElementById('trade-qty'),
    tradeTotal: document.getElementById('trade-total'),
    modalClose: document.getElementById('modal-close'),
    qtyMinus: document.getElementById('qty-minus'),
    qtyPlus: document.getElementById('qty-plus'),
    btnBuy: document.getElementById('btn-buy'),
    btnSell: document.getElementById('btn-sell'),

    resultBadge: document.getElementById('result-badge'),
    resultText: document.getElementById('result-text'),
    finalYourProfit: document.getElementById('final-your-profit'),
    finalCpuProfit: document.getElementById('final-cpu-profit'),
    matchStats: document.getElementById('match-stats')
};

// ===== UTILITY FUNCTIONS =====
const formatCurrency = (amount) => {
    const sign = amount >= 0 ? '' : '-';
    return `${sign}₹${Math.abs(Math.round(amount)).toLocaleString()}`;
};

const formatOvers = (balls) => {
    const overs = Math.floor(balls / 6);
    const ballsInOver = balls % 6;
    return `${overs}.${ballsInOver}`;
};

const getOutcomeText = (outcome) => {
    const texts = {
        0: ['DOT BALL!', 'No run, good bowling!', 'Defended well!'],
        1: ['SINGLE!', 'Quick single taken!', 'Easy run!'],
        2: ['TWO RUNS!', 'Great running!', 'Pushed for two!'],
        3: ['THREE RUNS!', 'Excellent running!', 'All run three!'],
        4: ['FOUR!', 'BOUNDARY!', 'Cracking shot!', 'Races to the fence!'],
        6: ['SIX!', 'MAXIMUM!', 'OUT OF THE PARK!', 'HUGE SIX!'],
        'W': ['WICKET!', 'OUT!', 'Bowled him!', 'Gone!']
    };
    const options = texts[outcome];
    return options[Math.floor(Math.random() * options.length)];
};

const getOutcomeEmoji = (outcome) => {
    const emojis = {
        0: '⚫',
        1: '🏃',
        2: '🏃‍♂️',
        3: '🏃‍♀️',
        4: '4️⃣',
        6: '🔥',
        'W': '💀'
    };
    return emojis[outcome] || '🏏';
};

// ===== TRADING FUNCTIONS =====
const buyStock = (playerId, quantity, isCpu = false) => {
    const player = gameState.players.find(p => p.id === playerId);
    if (!player) return false;

    const totalCost = player.price * quantity;
    const portfolio = isCpu ? gameState.cpuPortfolio : gameState.yourPortfolio;
    const cash = isCpu ? gameState.cpuCash : gameState.yourCash;

    if (totalCost > cash) return false;

    if (!portfolio[playerId]) {
        portfolio[playerId] = { quantity: 0, avgPrice: 0, totalCost: 0 };
    }

    const holding = portfolio[playerId];
    holding.totalCost += totalCost;
    holding.quantity += quantity;
    holding.avgPrice = holding.totalCost / holding.quantity;

    if (isCpu) {
        gameState.cpuCash -= totalCost;
        gameState.cpuTrades++;
    } else {
        gameState.yourCash -= totalCost;
        gameState.totalTrades++;
    }

    return true;
};

const sellStock = (playerId, quantity, isCpu = false) => {
    const player = gameState.players.find(p => p.id === playerId);
    if (!player) return false;

    const portfolio = isCpu ? gameState.cpuPortfolio : gameState.yourPortfolio;
    const holding = portfolio[playerId];

    if (!holding || holding.quantity < quantity) return false;

    const saleValue = player.price * quantity;
    const costBasis = holding.avgPrice * quantity;
    const realizedProfit = saleValue - costBasis;

    holding.quantity -= quantity;
    holding.totalCost = holding.avgPrice * holding.quantity;

    if (holding.quantity === 0) {
        delete portfolio[playerId];
    }

    if (isCpu) {
        gameState.cpuCash += saleValue;
        gameState.cpuRealizedProfit += realizedProfit;
        gameState.cpuTrades++;
    } else {
        gameState.yourCash += saleValue;
        gameState.yourRealizedProfit += realizedProfit;
        gameState.totalTrades++;
    }

    return true;
};

const calculateProfit = (isCpu = false) => {
    const portfolio = isCpu ? gameState.cpuPortfolio : gameState.yourPortfolio;
    const realizedProfit = isCpu ? gameState.cpuRealizedProfit : gameState.yourRealizedProfit;

    let unrealizedProfit = 0;

    Object.entries(portfolio).forEach(([playerId, holding]) => {
        const player = gameState.players.find(p => p.id === playerId);
        if (player) {
            const currentValue = player.price * holding.quantity;
            unrealizedProfit += currentValue - holding.totalCost;
        }
    });

    return realizedProfit + unrealizedProfit;
};

// ===== SMARTER CPU AI TRADING =====
const cpuTrade = () => {
    // CPU uses momentum and mean reversion strategy
    const rand = Math.random();
    if (rand < 0.25) return; // CPU sometimes does nothing

    gameState.players.forEach(player => {
        const history = player.priceHistory;
        if (history.length < 3) return;

        const momentum = calculateMomentum(player);
        const priceRatio = player.price / player.basePrice;
        const holding = gameState.cpuPortfolio[player.id];

        // Buy on positive momentum or if undervalued
        if ((momentum > 0.05 || priceRatio < 0.85) && gameState.cpuCash >= player.price * 2 && Math.random() < 0.35) {
            const qty = Math.floor(Math.random() * 3) + 1;
            buyStock(player.id, qty, true);
        }
        // Sell on negative momentum or if overvalued
        else if ((momentum < -0.05 || priceRatio > 1.3) && holding && holding.quantity > 0 && Math.random() < 0.4) {
            const qty = Math.min(holding.quantity, Math.floor(Math.random() * 2) + 1);
            sellStock(player.id, qty, true);
        }
        // Take profits if up significantly
        else if (holding && holding.quantity > 0) {
            const profitPercent = (player.price - holding.avgPrice) / holding.avgPrice;
            if (profitPercent > 0.12 && Math.random() < 0.25) {
                sellStock(player.id, 1, true);
            }
        }
    });
};

// ===== UI RENDERING =====
const renderStocks = () => {
    elements.stocksGrid.innerHTML = gameState.players.map(player => {
        const history = player.priceHistory;
        const change = history.length > 1
            ? ((player.price - history[history.length - 2]) / history[history.length - 2]) * 100
            : 0;
        const changeClass = change >= 0 ? 'positive' : 'negative';
        const cardClass = Math.abs(change) > 3 ? (change >= 0 ? 'up' : 'down') : '';

        const momentum = calculateMomentum(player);
        const momentumIndicator = momentum > 0.03 ? '📈' : (momentum < -0.03 ? '📉' : '➡️');

        const sparkline = createSparkline(history);

        return `
            <div class="stock-card ${cardClass}" data-player-id="${player.id}">
                <div class="stock-header">
                    <span class="stock-name">${player.name}</span>
                    <span class="stock-role ${player.role}">${player.role === 'batsman' ? '🏏' : '🎳'}</span>
                </div>
                <div class="stock-price">${formatCurrency(player.price)}</div>
                <div class="stock-change ${changeClass}">
                    ${change >= 0 ? '▲' : '▼'} ${Math.abs(change).toFixed(1)}% ${momentumIndicator}
                </div>
                <div class="stock-chart">${sparkline}</div>
                <div class="stock-actions">
                    <button class="stock-btn buy" onclick="openTradeModal('${player.id}')">BUY</button>
                    <button class="stock-btn sell" onclick="openTradeModal('${player.id}')">SELL</button>
                </div>
            </div>
        `;
    }).join('');
};

const createSparkline = (data) => {
    if (data.length < 2) return '<svg></svg>';

    const width = 150;
    const height = 40;
    const padding = 4;

    const min = Math.min(...data);
    const max = Math.max(...data);
    const range = max - min || 1;

    const points = data.map((val, i) => {
        const x = padding + (i / (data.length - 1)) * (width - padding * 2);
        const y = height - padding - ((val - min) / range) * (height - padding * 2);
        return `${x},${y}`;
    }).join(' ');

    const lastVal = data[data.length - 1];
    const prevVal = data[data.length - 2] || lastVal;
    const color = lastVal >= prevVal ? '#10b981' : '#ef4444';

    // Add area fill for smoother visual
    const areaPoints = `${padding},${height - padding} ${points} ${width - padding},${height - padding}`;

    return `
        <svg viewBox="0 0 ${width} ${height}">
            <polygon fill="${color}22" points="${areaPoints}"/>
            <polyline fill="none" stroke="${color}" stroke-width="2" points="${points}"/>
        </svg>
    `;
};

const renderPortfolio = () => {
    const entries = Object.entries(gameState.yourPortfolio);

    if (entries.length === 0) {
        elements.portfolioList.innerHTML = '<div class="portfolio-empty">No holdings yet. Buy some stocks!</div>';
    } else {
        elements.portfolioList.innerHTML = entries.map(([playerId, holding]) => {
            const player = gameState.players.find(p => p.id === playerId);
            if (!player) return '';

            const currentValue = player.price * holding.quantity;
            const pnl = currentValue - holding.totalCost;
            const pnlClass = pnl >= 0 ? 'positive' : 'negative';

            return `
                <div class="portfolio-item">
                    <div class="portfolio-item-header">
                        <span class="portfolio-item-name">${player.name}</span>
                        <span class="portfolio-item-qty">${holding.quantity} @ ₹${Math.round(holding.avgPrice)}</span>
                    </div>
                    <div class="portfolio-item-pnl ${pnlClass}">
                        ${formatCurrency(pnl)} (${((pnl / holding.totalCost) * 100).toFixed(1)}%)
                    </div>
                </div>
            `;
        }).join('');
    }

    let totalInvested = 0;
    let currentValue = 0;

    entries.forEach(([playerId, holding]) => {
        const player = gameState.players.find(p => p.id === playerId);
        if (player) {
            totalInvested += holding.totalCost;
            currentValue += player.price * holding.quantity;
        }
    });

    const unrealizedPnl = currentValue - totalInvested;

    elements.totalInvested.textContent = formatCurrency(totalInvested);
    elements.currentValue.textContent = formatCurrency(currentValue);
    elements.unrealizedPnl.textContent = formatCurrency(unrealizedPnl);
    elements.unrealizedPnl.className = unrealizedPnl >= 0 ? 'positive' : 'negative';
};

const updateProfitDisplay = () => {
    gameState.yourProfit = calculateProfit(false);
    gameState.cpuProfit = calculateProfit(true);

    elements.yourProfit.textContent = formatCurrency(gameState.yourProfit);
    elements.cpuProfit.textContent = formatCurrency(gameState.cpuProfit);

    const totalProfit = Math.abs(gameState.yourProfit) + Math.abs(gameState.cpuProfit);
    if (totalProfit > 0) {
        const diff = gameState.yourProfit - gameState.cpuProfit;
        const percentage = Math.min(50, Math.abs(diff / totalProfit) * 50);

        if (diff > 0) {
            elements.indicatorFill.style.left = '50%';
            elements.indicatorFill.style.width = `${percentage}%`;
            elements.indicatorFill.style.background = 'var(--accent-green)';
        } else {
            elements.indicatorFill.style.left = `${50 - percentage}%`;
            elements.indicatorFill.style.width = `${percentage}%`;
            elements.indicatorFill.style.background = 'var(--accent-red)';
        }
    }
};

const updateMatchDisplay = () => {
    elements.score.textContent = `${gameState.runs}/${gameState.wickets}`;
    elements.overs.textContent = formatOvers(gameState.balls);

    const runRate = gameState.balls > 0 ? ((gameState.runs / gameState.balls) * 6).toFixed(2) : '0.00';
    elements.runRate.textContent = runRate;

    if (gameState.innings === 2 && gameState.target) {
        const ballsRemaining = 120 - gameState.balls;
        const runsNeeded = gameState.target - gameState.runs;
        const reqRate = ballsRemaining > 0 ? ((runsNeeded / ballsRemaining) * 6).toFixed(2) : '-';
        elements.requiredRate.textContent = reqRate;
    } else {
        elements.requiredRate.textContent = '-';
    }

    const striker = gameState.players[gameState.strikerIndex];
    const bowler = gameState.players[gameState.currentBowlerIndex + 4];

    if (striker) {
        elements.strikerName.textContent = striker.name;
        elements.strikerScore.textContent = `${striker.runs}(${striker.balls})`;
    }

    if (bowler) {
        elements.bowlerName.textContent = bowler.name;
        elements.bowlerFigures.textContent = `${bowler.wickets}/${bowler.runsConceded}`;
    }

    elements.yourCash.textContent = formatCurrency(gameState.yourCash);
};

const addCommentary = (arg1, arg2, arg3) => {
    // Case 1: System Message (String)
    if (typeof arg1 === 'string') {
        const text = arg1;
        const type = arg2 || '';
        const p = document.createElement('div');
        p.className = `commentary-item ${type}`;
        p.textContent = text;
        elements.commentary.prepend(p);
    }
    // Case 2: Ball Outcome (AI or Fallback)
    else {
        const outcome = arg1;
        const batsman = arg2;
        const bowler = arg3;

        if (gemini) {
            const commentaryContext = {
                team1: 'IND',
                team2: 'AUS',
                score: gameState.runs,
                wickets: gameState.wickets,
                overs: formatOvers(gameState.balls),
                batsman: batsman.name,
                batsmanRuns: batsman.runs,
                batsmanBalls: batsman.balls,
                bowler: bowler.name,
                bowlerWickets: bowler.wickets,
                bowlerRuns: bowler.runsConceded,
                outcome: outcome === 'W' ? 'WICKET' : (outcome === 0 ? 'DOT' : (outcome === 4 ? 'FOUR' : (outcome === 6 ? 'SIX' : (outcome === 1 ? 'SINGLE' : (outcome === 2 ? 'DOUBLE' : 'TRIPLE'))))),
                phase: gameState.balls < 36 ? 'Powerplay' : (gameState.balls >= 90 ? 'Death Overs' : 'Middle Overs')
            };

            gemini.generateCommentary(commentaryContext).then(text => {
                const p = document.createElement('div');
                p.className = `commentary-item ${outcome === 'W' ? 'wicket' : (outcome === 4 ? 'four' : (outcome === 6 ? 'six' : ''))}`;
                p.innerHTML = `<span style="font-size:0.8em">✨</span> ${text}`;
                elements.commentary.prepend(p);
                if (elements.commentary.children.length > 20) elements.commentary.lastChild.remove();
            });
        } else {
            // Fallback
            const p = document.createElement('div');
            p.className = `commentary-item ${outcome === 'W' ? 'wicket' : (outcome === 4 ? 'four' : (outcome === 6 ? 'six' : ''))}`;
            let text = `${batsman.name}: ${getOutcomeText(outcome)}`;
            if (outcome === 'W') text += ` b ${bowler.name}`;
            p.textContent = text;
            elements.commentary.prepend(p);
        }
    }

    // Cleanup
    while (elements.commentary.children.length > 20) {
        elements.commentary.lastChild.remove();
    }
};

const showBallResult = (outcome) => {
    const emoji = getOutcomeEmoji(outcome);
    const text = getOutcomeText(outcome);

    elements.ballResult.innerHTML = `
        <div class="result-icon">${emoji}</div>
        <div class="result-text">${text}</div>
    `;

    elements.ballResult.classList.add(outcome === 'W' ? 'flash-down' : (outcome >= 4 ? 'flash-up' : ''));
    setTimeout(() => {
        elements.ballResult.classList.remove('flash-up', 'flash-down');
    }, 500);
};

// ===== TRADE MODAL =====
const openTradeModal = (playerId) => {
    const player = gameState.players.find(p => p.id === playerId);
    if (!player) return;

    gameState.selectedPlayer = player;

    const history = player.priceHistory;
    const change = history.length > 1
        ? ((player.price - history[history.length - 2]) / history[history.length - 2]) * 100
        : 0;
    const changeClass = change >= 0 ? 'positive' : 'negative';

    const holding = gameState.yourPortfolio[playerId];
    const qty = holding ? holding.quantity : 0;

    elements.modalPlayerName.textContent = player.name;
    elements.modalPrice.textContent = formatCurrency(player.price);
    elements.modalChange.textContent = `${change >= 0 ? '+' : ''}${change.toFixed(1)}%`;
    elements.modalChange.className = `change ${changeClass}`;
    elements.modalHoldings.textContent = qty.toString();
    elements.tradeQty.value = 1;
    elements.tradeTotal.textContent = formatCurrency(player.price);

    elements.modal.classList.add('active');
};

const closeTradeModal = () => {
    elements.modal.classList.remove('active');
    gameState.selectedPlayer = null;
};

const updateTradeTotal = () => {
    if (!gameState.selectedPlayer) return;
    const qty = parseInt(elements.tradeQty.value) || 1;
    elements.tradeTotal.textContent = formatCurrency(gameState.selectedPlayer.price * qty);
};

// Modal event listeners
elements.modalClose.addEventListener('click', closeTradeModal);
elements.modal.addEventListener('click', (e) => {
    if (e.target === elements.modal) closeTradeModal();
});

elements.qtyMinus.addEventListener('click', () => {
    let qty = parseInt(elements.tradeQty.value) || 1;
    if (qty > 1) {
        elements.tradeQty.value = qty - 1;
        updateTradeTotal();
    }
});

elements.qtyPlus.addEventListener('click', () => {
    let qty = parseInt(elements.tradeQty.value) || 1;
    if (qty < 100) {
        elements.tradeQty.value = qty + 1;
        updateTradeTotal();
    }
});

elements.tradeQty.addEventListener('input', updateTradeTotal);

elements.btnBuy.addEventListener('click', () => {
    if (!gameState.selectedPlayer) return;
    const qty = parseInt(elements.tradeQty.value) || 1;

    if (buyStock(gameState.selectedPlayer.id, qty, false)) {
        renderPortfolio();
        updateProfitDisplay();
        updateMatchDisplay();
        closeTradeModal();
    } else {
        alert('Not enough cash!');
    }
});

elements.btnSell.addEventListener('click', () => {
    if (!gameState.selectedPlayer) return;
    const qty = parseInt(elements.tradeQty.value) || 1;

    if (sellStock(gameState.selectedPlayer.id, qty, false)) {
        renderPortfolio();
        updateProfitDisplay();
        updateMatchDisplay();
        closeTradeModal();
    } else {
        alert('Not enough holdings to sell!');
    }
});

window.openTradeModal = openTradeModal;

// ===== GAME LOGIC =====
const simulateBall = () => {
    if (!gameState.isRunning) return;

    const batsmanIndex = gameState.strikerIndex;

    // Determine Bowler Index
    // Bowling team starts at 0 (RCB) or 11 (CSK)
    const bowlingTeamStart = gameState.bowlingTeam === 'RCB' ? 0 : 11;
    // Use simplest rotation: 5 bowlers (indices 6,7,8,9,10 relative to start)
    // Map currentBowlerIndex (0-4) to actual player index
    // e.g. Bowler 1 is at index start+6, Bowler 2 at start+7...
    const bowlerRelativeIndex = 6 + (gameState.currentBowlerIndex % 5);
    const bowlerIndex = bowlingTeamStart + bowlerRelativeIndex;

    const batsman = gameState.players[batsmanIndex];
    const bowler = gameState.players[bowlerIndex];

    // DATA-DRIVEN OUTCOME based on real player stats
    const outcome = getDataDrivenOutcome(batsman, bowler, gameState.balls);

    gameState.balls++;

    if (outcome === 'W') {
        gameState.wickets++;
        batsman.isOut = true;
        bowler.wickets++;
        addCommentary(outcome, batsman, bowler);

        if (gameState.wickets < 10) {
            const nextIdx = getNextBatsmanIndex();
            if (nextIdx !== -1) {
                gameState.strikerIndex = nextIdx;
                addCommentary(`New Batter: ${gameState.players[nextIdx].name}`, 'info');
            } else {
                // All out
                addCommentary("ALL OUT!", "wicket");
            }
        }
    } else {
        const runs = outcome;
        gameState.runs += runs;
        batsman.runs += runs;
        batsman.balls++;
        bowler.runsConceded += runs;

        // Unified call for all run outcomes
        addCommentary(outcome, batsman, bowler);

        if (runs % 2 === 1) {
            [gameState.strikerIndex, gameState.nonStrikerIndex] =
                [gameState.nonStrikerIndex, gameState.strikerIndex];
        }
    }

    // Update prices using GBM
    updatePrices(outcome, batsmanIndex, bowlerIndex);

    showBallResult(outcome);

    // Change bowler every over
    if (gameState.balls % 6 === 0) {
        gameState.currentBowlerIndex = (gameState.currentBowlerIndex + 1) % 5;
        // Strike rotation at over end
        [gameState.strikerIndex, gameState.nonStrikerIndex] =
            [gameState.nonStrikerIndex, gameState.strikerIndex];
    }

    // CPU trading with momentum strategy
    cpuTrade();

    // Update UI
    renderStocks();
    renderPortfolio();
    updateMatchDisplay();
    updateProfitDisplay();

    // Check innings/match end
    if (gameState.wickets >= 10 || gameState.balls >= 120) {
        if (gameState.innings === 1) {
            endFirstInnings();
        } else {
            endMatch(gameState.runs >= gameState.target ? gameState.battingTeam : gameState.bowlingTeam);
        }
    } else if (gameState.innings === 2 && gameState.runs >= gameState.target) {
        endMatch(gameState.battingTeam);
    }
};

const getNextBatsmanIndex = () => {
    // Find first player in batting team who is not out and not currently batting
    const teamPrefix = gameState.battingTeam; // 'RCB' or 'CSK'

    // Players are stored in order: RCB (0-10), CSK (11-21)
    let startIndex = teamPrefix === 'RCB' ? 0 : 11;
    let endIndex = teamPrefix === 'RCB' ? 10 : 21;

    for (let i = startIndex; i <= endIndex; i++) {
        const player = gameState.players[i];
        if (!player.isOut && i !== gameState.strikerIndex && i !== gameState.nonStrikerIndex) {
            return i;
        }
    }
    return -1; // All out
};

const endFirstInnings = () => {
    gameState.firstInningsScore = gameState.runs;
    gameState.target = gameState.runs + 1;

    // Swap Innings
    gameState.innings = 2;
    gameState.runs = 0;
    gameState.wickets = 0;
    gameState.balls = 0;

    // Swap Teams
    const temp = gameState.battingTeam;
    gameState.battingTeam = gameState.bowlingTeam;
    gameState.bowlingTeam = temp;

    // Reset Batting Indices
    // RCB starts at 0, CSK starts at 11
    const battingStart = gameState.battingTeam === 'RCB' ? 0 : 11;
    gameState.strikerIndex = battingStart;
    gameState.nonStrikerIndex = battingStart + 1;

    // Reset Bowler
    gameState.currentBowlerIndex = 0; // Will be mapped to actual index in simulateBall

    elements.matchStage.textContent = `2nd Innings - ${gameState.battingTeam} needs ${gameState.target}`;
    addCommentary(`Innings Break! target set: ${gameState.target}. ${gameState.battingTeam} coming out to bat.`);
};

const endMatch = (winnerTeam) => {
    gameState.isRunning = false;
    clearInterval(gameState.timerInterval);

    // Liquidate all
    Object.keys(gameState.yourPortfolio).forEach(playerId => {
        const holding = gameState.yourPortfolio[playerId];
        if (holding && holding.quantity > 0) sellStock(playerId, holding.quantity, false);
    });
    Object.keys(gameState.cpuPortfolio).forEach(playerId => {
        const holding = gameState.cpuPortfolio[playerId];
        if (holding && holding.quantity > 0) sellStock(playerId, holding.quantity, true);
    });

    showEndScreen(gameState.yourRealizedProfit, gameState.cpuRealizedProfit, winnerTeam);
};

const showEndScreen = (yourProfit, cpuProfit, winnerTeam) => {
    elements.gameScreen.classList.remove('active');
    elements.endScreen.classList.add('active');

    elements.finalYourProfit.textContent = formatCurrency(yourProfit);
    elements.finalCpuProfit.textContent = formatCurrency(cpuProfit);

    let resultMsg = '';
    // Trading Result
    if (yourProfit > cpuProfit) {
        elements.resultBadge.className = 'result-badge win';
        elements.resultBadge.querySelector('.result-icon').textContent = '🏆';
        resultMsg = 'YOU WON THE TRADING BATTLE!';
    } else if (cpuProfit > yourProfit) {
        elements.resultBadge.className = 'result-badge lose';
        elements.resultBadge.querySelector('.result-icon').textContent = '😢';
        resultMsg = 'CPU WON THE TRADING BATTLE!';
    } else {
        elements.resultBadge.className = 'result-badge draw';
        elements.resultBadge.querySelector('.result-icon').textContent = '🤝';
        resultMsg = "TRADING DRAW!";
    }

    // Match Result
    const matchResult = winnerTeam ? `${winnerTeam} won the match!` : "Match Abandoned";
    elements.resultText.innerHTML = `${resultMsg}<br><span style="font-size:0.8em; color:#ccc">${matchResult}</span>`;

    elements.matchStats.innerHTML = `
        <div class="stat-item"><span class="label">Your Trades:</span><span class="value">${gameState.totalTrades}</span></div>
        <div class="stat-item"><span class="label">CPU Trades:</span><span class="value">${gameState.cpuTrades}</span></div>
        <div class="stat-item"><span class="label">Total Sixes:</span><span class="value">${gameState.sixCount}</span></div>
        <div class="stat-item"><span class="label">Total Fours:</span><span class="value">${gameState.fourCount}</span></div>
        <div class="stat-item"><span class="label">Final Score:</span><span class="value">${gameState.runs}/${gameState.wickets}</span></div>
        <div class="stat-item"><span class="label">1st Innings:</span><span class="value">${gameState.firstInningsScore}</span></div>
    `;
};

const startTimer = () => {
    gameState.ballTimer = 5;
    elements.timer.textContent = gameState.ballTimer;

    clearInterval(gameState.timerInterval);
    gameState.timerInterval = setInterval(() => {
        if (!gameState.isRunning) {
            clearInterval(gameState.timerInterval);
            return;
        }

        gameState.ballTimer--;
        elements.timer.textContent = Math.max(0, gameState.ballTimer);

        if (gameState.ballTimer <= 0) {
            gameState.ballTimer = 5;
            elements.timer.textContent = gameState.ballTimer;
            simulateBall();
        }
    }, 1000);
};

const startGame = () => {
    // Reset game state
    gameState.innings = 1;
    gameState.battingTeam = 'RCB';
    gameState.bowlingTeam = 'CSK';
    gameState.runs = 0;
    gameState.wickets = 0;
    gameState.balls = 0;
    gameState.target = null;
    gameState.firstInningsScore = 0;

    gameState.yourCash = 10000;
    gameState.yourPortfolio = {};
    gameState.yourProfit = 0;
    gameState.yourRealizedProfit = 0;

    gameState.cpuCash = 10000;
    gameState.cpuPortfolio = {};
    gameState.cpuProfit = 0;
    gameState.cpuRealizedProfit = 0;

    // Initialize Players
    gameState.players = createPlayers();

    // Set Initial Strikers (RCB starts at 0)
    gameState.strikerIndex = 0;
    gameState.nonStrikerIndex = 1;
    gameState.currentBowlerIndex = 0;

    gameState.totalTrades = 0;
    gameState.cpuTrades = 0;
    gameState.sixCount = 0;
    gameState.fourCount = 0;

    gameState.isRunning = true;

    elements.matchStage.textContent = '1st Innings - RCB Batting';
    elements.startScreen.classList.remove('active');
    elements.endScreen.classList.remove('active');
    elements.gameScreen.classList.add('active');

    elements.commentary.innerHTML = '<div class="commentary-item">Match Started! RCB vs CSK. 11 Players each side.</div>';

    renderStocks();
    renderPortfolio();
    updateMatchDisplay();
    updateProfitDisplay();

    startTimer();
};

// ===== EVENT LISTENERS =====
elements.startBtn.addEventListener('click', startGame);
elements.playAgainBtn.addEventListener('click', startGame);

// Initial render
elements.startScreen.classList.add('active');

console.log('🏏 Cricket Trading Battle - Realistic Simulation Loaded');
console.log('📊 Using real IPL player statistics');
console.log('📈 GBM price engine with momentum indicators');
