require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
const port = process.env.PORT || 3000;

// Initialize Gemini API
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

app.use(cors());
// Increase payload limit for base64 images.
// The app downscales to a 1024px longest edge before upload (see ImageDownscaler), so a
// real request is well under 1MB. This ceiling only exists for older clients.
app.use(express.json({ limit: '12mb' }));

// ---------------------------------------------------------------------------
// Optional shared-secret auth.
//
// Off unless SCAN_API_KEY is set, because v1.9.x is live in production and does not send
// a key - turning this on unconditionally would break every installed copy of the app.
// Set the variable only once a release that sends the header has rolled out.
// ---------------------------------------------------------------------------
const SCAN_API_KEY = process.env.SCAN_API_KEY || null;

function requireKey(req, res, next) {
    if (!SCAN_API_KEY) return next();
    if (req.get('x-api-key') === SCAN_API_KEY) return next();
    return res.status(401).json({ error: 'unauthorized', code: 'UNAUTHORIZED' });
}

// ---------------------------------------------------------------------------
// Rate limiting, in memory.
//
// Deliberately not a dependency: one Render instance, one process, and the point is only
// to stop an open endpoint from running up a Gemini bill. Counters reset on restart,
// which is acceptable for that purpose.
// ---------------------------------------------------------------------------
const RATE_LIMIT_WINDOW_MS = 60 * 1000;
const RATE_LIMIT_MAX = Number(process.env.RATE_LIMIT_MAX || 20);
const hits = new Map();

function rateLimit(req, res, next) {
    const now = Date.now();
    const ip = req.ip || req.socket.remoteAddress || 'unknown';
    const entry = hits.get(ip);

    if (!entry || now - entry.start > RATE_LIMIT_WINDOW_MS) {
        hits.set(ip, { start: now, count: 1 });
    } else if (entry.count >= RATE_LIMIT_MAX) {
        return res.status(429).json({ error: 'too many requests', code: 'RATE_LIMITED' });
    } else {
        entry.count += 1;
    }

    // Bound the map so a spray of addresses cannot grow it without limit.
    if (hits.size > 5000) {
        for (const [key, value] of hits) {
            if (now - value.start > RATE_LIMIT_WINDOW_MS) hits.delete(key);
        }
    }
    next();
}

// ---------------------------------------------------------------------------
// The recognition prompt.
//
// This previously restricted Gemini to seven words:
//
//     ONLY identify items found in this exact list:
//     ["Rice", "Roti", "Dal", "Paneer", "Curd", "Sabzi", "Potato"]
//
// which meant the service could not report "Dal makhani" or "Masala dosa" however clearly
// they appeared. Every thali came back as some subset of seven generic words, and the app
// - which now carries a 768-dish nutrition table - had nothing specific to look up. A
// four-dish plate was reported as two.
//
// Gemini now names the dish it actually sees and says how many. Two rules matter:
//
//  1. Name the dish, do not judge it. Nutrition is computed on the device against a known
//     table, never taken from the model. That separation is deliberate: it keeps results
//     reproducible and explainable, and it is what lets the app state a figure as an
//     estimate it can account for.
//  2. Say only what is visible. A model that guesses at a hidden filling produces a number
//     the user cannot check, and an unrecognised dish handled honestly is better than a
//     confident wrong one.
// ---------------------------------------------------------------------------
const PROMPT_FREEFORM = `
You are identifying the dishes in a photograph of a meal, usually an Indian thali or plate.

THREE ABSOLUTE CONSTRAINTS. Apply these before anything else below.

A. NEVER NAME A CONTAINER YOU CANNOT SEE INTO.
   Look at each bowl, katori, glass and vessel. If it is cut off by the edge of the frame,
   hidden behind another dish, turned away, or its contents are otherwise not clearly
   visible, LEAVE IT OUT ENTIRELY. Do not name it from its shape, its size, its position
   on the plate, its lid, or from what usually accompanies this meal. A partly visible rim
   with nothing identifiable inside it is not a dish - it is a container, and you omit it.
   This applies no matter how confident you feel about what is probably in it.

B. COUNT WHOLE ITEMS, NEVER PIECES.
   Before giving any count above 1, ask yourself: am I counting separate items, or parts
   of one item? A bread folded in half or quarters is ONE bread. A papad broken into
   fragments is ONE papad. A stack seen edge-on shows more folds than there are breads.
   If the count is not obvious, give the LOWER number and set confidence "low".

C. ONE ENTRY PER PHYSICAL SERVING.
   Every entry in your answer must be a different portion of food on the plate. Before you
   add an entry, check it against the ones you already have: is this the same food I have
   already named? If yes, do not add it again.
   The mistake this prevents is naming one food twice - once specifically and once
   generally. A single serving of peas pulao is "Peas pulao" and nothing else; it is NOT
   also "Boiled rice". Dal makhani is not also "Dal". Aloo gobi is not also "Mixed
   vegetable curry". Masala dosa is not also "Dosa". Pick the one name that best describes
   what you see, and use it once.
   Rice deserves particular care: pulao, biryani, jeera rice, curd rice and plain rice are
   alternatives, not companions. Report plain rice alongside a flavoured rice only when you
   can see two visibly separate rice portions on the plate.

Everything you report must be something you can point to in this photograph. A dish you
leave out, the person can add. A dish you invent becomes a number they never ate, and
nothing on their screen will tell them it was a guess.

Return a JSON array. Each element describes one dish you can actually see:

  {"name": "<dish name>", "quantity": <number>, "confidence": "high"|"medium"|"low"}

RULES

1. Name the dish as specifically as the photo supports. Prefer "Dal makhani" over "Dal",
   "Masala dosa" over "Dosa", "Aloo paratha" over "Paratha". If you can only tell it is a
   dal and not which dal, then "Dal" is the correct answer - do not invent detail.

2. Use the common Indian name for the dish. English is fine where that is what people say
   ("Boiled rice", "Mixed vegetable curry"). Do not translate a name nobody uses.

3. quantity is how many WHOLE SEPARATE items are on the plate.

   Count whole items, not pieces of them:
   - A flatbread folded in half, in quarters, or rolled up is ONE bread, not two or four.
     Count the breads, not the visible folded edges. A stack of folded rotis is easy to
     over-count; if you cannot see clearly how many there are, give your lower estimate
     and set confidence "low".
   - A papad or biscuit that has cracked or broken is ONE item, not one per fragment.
   - Anything served in a bowl, katori, glass or as a scooped portion is 1, whatever its
     size: dal, rice, curd, curry, raita, sabzi. A large mound of rice is still 1. Never
     report 1.5 or 2 for a single scooped serving because it looks generous - the size of a
     portion is not a count, and the app already assumes a full serving.

   Use 0.5 only for an obvious half portion. Never guess above 12. When in doubt about a
   count, prefer the LOWER number: over-counting a staple adds calories the person never
   ate, which is worse than missing one.

4. confidence is "high" when the dish is unmistakable, "medium" when the category is clear
   but the exact preparation is not, "low" when you are unsure. Prefer "medium" or "low"
   over omitting a dish you can see. If you are confident about the dish but not its
   count, still use "low".

5. Constraint A again, because it is the one most often broken: do not name a container
   whose contents you cannot see. Also do not infer a filling or stuffing you cannot see,
   do not add a dish because it commonly comes with the others, and do not include
   cutlery, napkins, garnish, plain water, or empty vessels.

6. Constraint C again: read your finished list before returning it. If any two entries
   could be the same portion of food under two different names, delete the vaguer one and
   keep the specific one. Two entries naming the same serving double its calories, and the
   person has no way to know which of the two is the invented one.

7. Return ONLY the JSON array. No markdown, no backticks, no commentary. An empty plate,
   or a photo that is not food, returns [].

EXAMPLE - a plate with one flavoured rice, note that plain rice is NOT also listed
[{"name":"Chapati","quantity":3,"confidence":"high"},
 {"name":"Dal makhani","quantity":1,"confidence":"medium"},
 {"name":"Peas pulao","quantity":1,"confidence":"high"},
 {"name":"Papad","quantity":1,"confidence":"high"},
 {"name":"Curd","quantity":1,"confidence":"high"}]
`;

/**
 * The original seven-bucket prompt, kept verbatim for clients that cannot handle anything
 * else.
 *
 * Version 1.9.1 is live on Play and matches dish names against a hardcoded seven-food
 * table. Serving it "Dal makhani" would match nothing and every scan would fail with "no
 * food detected" - the whole installed base broken by a server-side change they never
 * asked for and cannot opt out of.
 *
 * So the free-form prompt is only used for clients that say they can handle it. Delete
 * this once Play Console shows no meaningful traffic on versions below the first release
 * that sends the header.
 */
const PROMPT_BUCKETS = `
    Analyze this Indian Thali image. Identify the food items for a metabolic health analysis.

    CRITICAL RULES:
    1. ONLY identify items found in this exact list: ["Rice", "Roti", "Dal", "Paneer", "Curd", "Sabzi", "Potato"].
    2. Map unknown items to the closest match (e.g., "Veg Curry" -> "Sabzi").
    3. Return ONLY a strict JSON array of objects with a single "name" key. NO coordinates, NO boxes.
    4. NO markdown formatting, NO backticks, NO explanations.

    Example exact output: [{"name": "Rice"}, {"name": "Dal"}]
`;

/**
 * Which prompt this client can handle.
 *
 * Opt-in by header rather than opt-out: a client that says nothing is assumed to be an old
 * one. Being wrong in that direction costs a new client some precision; being wrong in the
 * other direction breaks a shipped app.
 */
function wantsFreeform(req) {
    const features = (req.get('x-client-features') || '').toLowerCase();
    return features.split(',').map(s => s.trim()).includes('freeform-dishes');
}

const MAX_ITEMS = 15;
const MAX_QUANTITY = 12;

/**
 * Validates and normalises what the model returned.
 *
 * The model is prompted, not constrained, so its output is treated as untrusted: anything
 * malformed is dropped rather than passed through to the app, which would either fail to
 * decode it or - worse - show a nonsense figure to the user.
 */
function sanitize(raw) {
    if (!Array.isArray(raw)) return [];

    const out = [];
    for (const item of raw) {
        if (!item || typeof item !== 'object') continue;

        const name = typeof item.name === 'string' ? item.name.trim() : '';
        if (!name || name.length > 80) continue;

        let quantity = Number(item.quantity);
        if (!Number.isFinite(quantity) || quantity <= 0) quantity = 1;
        if (quantity > MAX_QUANTITY) quantity = MAX_QUANTITY;
        quantity = Math.round(quantity * 2) / 2; // halves are meaningful, thirds are not

        const confidence = ['high', 'medium', 'low'].includes(item.confidence)
            ? item.confidence
            : 'medium';

        out.push({ name, quantity, confidence });
        if (out.length >= MAX_ITEMS) break;
    }
    return out;
}

app.post('/api/vision-scan', rateLimit, requireKey, async (req, res) => {
    try {
        const { imageBase64 } = req.body;

        if (!imageBase64) {
            return res.status(400).json({ error: 'No imageBase64 provided', code: 'NO_IMAGE' });
        }

        const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });

        let base64Data = imageBase64;
        if (base64Data.startsWith('data:image')) {
            base64Data = base64Data.split(',')[1];
        }

        const imagePart = {
            inlineData: { data: base64Data, mimeType: 'image/jpeg' }
        };

        const freeform = wantsFreeform(req);
        const prompt = freeform ? PROMPT_FREEFORM : PROMPT_BUCKETS;

        let result;
        try {
            result = await model.generateContent([prompt, imagePart]);
        } catch (err) {
            // A failure to reach Gemini is OUR problem, not a claim about the user's photo.
            // This used to return 200 [] - identical to "nothing recognised" - so during an
            // outage the app told users their perfectly good photo was unreadable.
            console.error('Gemini API call failed:', err && err.message ? err.message : err);
            return res.status(502).json({ error: 'recognition service unavailable', code: 'UPSTREAM_FAILED' });
        }

        if (!result || !result.response) {
            return res.status(502).json({ error: 'empty response from recognition service', code: 'UPSTREAM_EMPTY' });
        }

        let parsed;
        try {
            const textResponse = result.response.text().trim();
            // Dish names only. The image itself is never logged, and nothing identifying
            // about the user reaches this service in the first place.
            console.log('Gemini response:', textResponse.slice(0, 500));

            const cleanedText = textResponse.replace(/```json/g, '').replace(/```/g, '').trim();
            parsed = JSON.parse(cleanedText);
        } catch (parseError) {
            // Also our problem, and also previously indistinguishable from "no food found".
            console.error('Failed to parse JSON from Gemini:', parseError.message);
            return res.status(502).json({ error: 'unreadable response from recognition service', code: 'UPSTREAM_UNREADABLE' });
        }

        const items = sanitize(parsed);

        // A genuine empty array now means exactly one thing: the model looked and found no
        // food. That is the only case where telling the user to retake the photo is fair.
        //
        // Old clients get exactly the shape they have always parsed. They ignore unknown
        // keys in practice, but sending fields a shipped app was never tested against is
        // not a risk worth taking for no benefit.
        res.status(200).json(
            freeform ? items : items.map(({ name }) => ({ name }))
        );
        console.log(`Recognised ${items.length} item(s) [${freeform ? 'freeform' : 'buckets'}]`);

    } catch (error) {
        console.error('Unexpected error:', error && error.message ? error.message : error);
        res.status(500).json({ error: 'internal error', code: 'INTERNAL' });
    }
});

/** Liveness check, so Render and uptime monitors do not need to POST an image. */
app.get('/health', (req, res) => {
    res.status(200).json({ ok: true, model: 'gemini-2.5-flash' });
});

app.listen(port, '0.0.0.0', () => {
    console.log(`\nBharat Vitals vision server listening on 0.0.0.0:${port}`);
    console.log(`auth: ${SCAN_API_KEY ? 'enabled' : 'disabled'} | rate limit: ${RATE_LIMIT_MAX}/min\n`);
});
