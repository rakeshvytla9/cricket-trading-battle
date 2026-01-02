// ===== SUPABASE DATABASE CLIENT =====
// TODO: Replace these with your project details from https://supabase.com
const SUPABASE_URL = 'YOUR_SUPABASE_URL_HERE';
const SUPABASE_KEY = 'YOUR_SUPABASE_ANON_KEY_HERE';

let supabase = null;

const initSupabase = () => {
    if (typeof createClient !== 'undefined' && SUPABASE_URL !== 'YOUR_SUPABASE_URL_HERE') {
        supabase = createClient(SUPABASE_URL, SUPABASE_KEY);
        console.log('🗄️ Database Connected');
    } else {
        console.warn('⚠️ Supabase not configured. Match history will not be saved.');
    }
};

const saveMatchResult = async (result) => {
    if (!supabase) return;

    const { error } = await supabase
        .from('matches')
        .insert([
            {
                score: result.score,
                winner: result.winner,
                user_profit: result.userProfit,
                cpu_profit: result.cpuProfit,
                timestamp: new Date().toISOString()
            }
        ]);

    if (error) console.error('Error saving match:', error);
    else console.log('Match saved to database!');
};
