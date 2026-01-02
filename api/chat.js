export default async function handler(req, res) {
    if (req.method !== 'POST') {
        return res.status(405).json({ error: 'Method not allowed' });
    }

    const { prompt, context } = req.body;
    const apiKey = process.env.GEMINI_API_KEY;

    if (!apiKey) {
        return res.status(500).json({ error: 'Server Error: GEMINI_API_KEY is missing in Vercel settings.' });
    }

    try {
        const payload = {
            contents: [{
                parts: [{
                    text: `Context: ${JSON.stringify(context)}. User Prompt: ${prompt}`
                }]
            }]
        };

        // Use gemini-1.5-flash for better stability
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key=${apiKey}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
        });

        const data = await response.json();

        // Check for API errors
        if (data.error) {
            console.error('Gemini API Error:', data.error);
            return res.status(400).json({ error: `Gemini Error: ${data.error.message}` });
        }

        // Check for empty candidates (Safety filters)
        if (!data.candidates || data.candidates.length === 0) {
            console.warn('Gemini returned no candidates:', data);
            return res.status(200).json({
                text: "I couldn't generate a response (Safety Filter triggered). Try a different prompt."
            });
        }

        const text = data.candidates[0].content?.parts?.[0]?.text;

        if (!text) {
            return res.status(200).json({ text: "Empty response from AI." });
        }

        return res.status(200).json({ text });

    } catch (error) {
        console.error('Server Internal Error:', error);
        return res.status(500).json({ error: 'Internal Server Error during AI fetch.' });
    }
}
