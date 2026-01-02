/**
 * Gemini 3 API Integration (Server-Side Proxy Version)
 * Features: AI Commentary, Trading Assistant, Prediction Reasoning
 */

class GeminiAI {
    constructor() {
        this.endpoint = '/api/chat';
        this.enabled = true;
    }

    async generateContent(prompt, context = {}) {
        try {
            const response = await fetch(this.endpoint, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ prompt, context })
            });

            if (!response.ok) {
                const err = await response.json();
                console.error('API Error:', err);
                return { error: err.error || `Server Error: ${response.statusText}` };
            }

            const data = await response.json();

            // Handle error returned in successful 200 response (legacy)
            if (data.error) return { error: data.error };

            return {
                text: data.text,
                success: true
            };
        } catch (error) {
            console.error('Gemini Proxy Error:', error);
            return { error: error.message };
        }
    }

    // ===== AI COMMENTARY =====
    async generateCommentary(gameState) {
        const prompt = `Match: ${gameState.team1} vs ${gameState.team2}
Score: ${gameState.score}/${gameState.wickets} (${gameState.overs} overs)
Batsman: ${gameState.batsman} (${gameState.batsmanRuns} off ${gameState.batsmanBalls})
Bowler: ${gameState.bowler} (${gameState.bowlerWickets}/${gameState.bowlerRuns})
Last ball outcome: ${gameState.outcome}
Phase: ${gameState.phase}

Task: exciting cricket commentary (max 15 words) for this ball.`;

        const result = await this.generateContent(prompt, { type: 'commentary' });
        return result.text || this.getFallbackCommentary(gameState.outcome);
    }

    getFallbackCommentary(outcome) {
        const fallbacks = {
            'DOT': ['Defended solidly!', 'Good line, no run.', 'Dot ball.'],
            'SINGLE': ['Quick single taken!', 'Rotates the strike.', 'Easy single.'],
            'DOUBLE': ['Good running between the wickets!', 'Two runs added.'],
            'TRIPLE': ['Excellent running! Three runs.', 'They push for three!'],
            'FOUR': ['FOUR! Brilliant shot!', 'Raced away to the boundary!', 'Cracking shot for four!'],
            'SIX': ['SIX! Massive hit!', 'Into the stands!', 'That\'s gone all the way!'],
            'WICKET': ['OUT! Big wicket!', 'GONE! The batsman has to walk!', 'Breakthrough!']
        };
        const options = fallbacks[outcome] || [''];
        return options[Math.floor(Math.random() * options.length)];
    }

    // ===== TRADING ASSISTANT =====
    async getTradingAdvice(query, gameState, portfolio) {
        const prompt = `Current Match State:
- Score: ${gameState.score}/${gameState.wickets} (${gameState.overs} overs)
- Phase: ${gameState.phase}
- Player Stocks: ${Object.keys(portfolio.stocks || {}).length} active

User Question: "${query}"

Task: Expert trading advice (Buy/Sell/Hold). Max 30 words.`;

        const result = await this.generateContent(prompt, { type: 'trading_advice' });
        if (result.error) {
            return `⚠️ Error: ${result.error}`;
        }
        return result.text || "I'm analyzing the market data. Please try again in a moment.";
    }

    // ===== MATCH SUMMARY =====
    async generateMatchSummary(matchData) {
        const prompt = `Match Winner: ${matchData.winner}
Final Score: ${matchData.score}/${matchData.wickets}
Trading Winner: ${matchData.tradingWinner}

Task: Exciting match summary (3 sentences).`;

        const result = await this.generateContent(prompt, { type: 'summary' });
        return result.text || "What a match! Check the leaderboard for your ranking.";
    }
}

// Export for use in game.js
window.GeminiAI = GeminiAI;
