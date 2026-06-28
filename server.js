require('dotenv').config();
const express = require('express');
const cors = require('cors');
const { GoogleGenerativeAI } = require('@google/generative-ai');

const app = express();
const port = process.env.PORT || 3000;

// Initialize Gemini API
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);

app.use(cors());
// Increase payload limit for base64 images
app.use(express.json({ limit: '50mb' }));

app.post('/api/vision-scan', async (req, res) => {
    try {
        const { imageBase64 } = req.body;
        
        if (!imageBase64) {
            return res.status(400).json({ error: 'No imageBase64 provided' });
        }

        const model = genAI.getGenerativeModel({ model: 'gemini-2.5-flash' });
        
        const prompt = `
    Analyze this Indian Thali image. Identify the food items for a metabolic health analysis.

    CRITICAL RULES:
    1. ONLY identify items found in this exact list: ["Rice", "Roti", "Dal", "Paneer", "Curd", "Sabzi", "Potato"]. 
    2. Map unknown items to the closest match (e.g., "Veg Curry" -> "Sabzi").
    3. Return ONLY a strict JSON array of objects with a single "name" key. NO coordinates, NO boxes.
    4. NO markdown formatting, NO backticks, NO explanations.

    Example exact output: [{"name": "Rice"}, {"name": "Dal"}]
`;

        let base64Data = imageBase64;
        if (base64Data.startsWith('data:image')) {
            base64Data = base64Data.split(',')[1];
        }

        const imagePart = {
            inlineData: {
                data: base64Data,
                mimeType: "image/jpeg"
            }
        };

        const result = await model.generateContent([prompt, imagePart]).catch(err => {
            console.error('Gemini API call failed:', err);
            return null;
        });

        if (!result || !result.response) {
            return res.json([]);
        }
        
        let foodData = [];
        try {
            const textResponse = result.response.text().trim();
            console.log("Raw Gemini Response:", textResponse);

            // Remove markdown backticks if Gemini accidentally includes them
            const cleanedText = textResponse.replace(/```json/g, '').replace(/```/g, '').trim();
            foodData = JSON.parse(cleanedText);

        } catch (parseError) {
            console.error("❌ Failed to parse JSON from Gemini:", parseError.message);
            // Do NOT crash. Send empty data so the Android app can show the default UI.
            foodData = []; 
        }

        // Always send a response back to the phone to prevent timeouts!
        res.status(200).json(foodData);
        console.log("🚀 Clean response successfully sent back to Android!");

    } catch (error) {
        console.error("❌ Gemini API Error:", error.message || error);
        res.status(500).json([]);
    }
});

app.listen(port, '0.0.0.0', () => {
    console.log(`\n🚀 Bharat Vitals Cloud Vision Server is awake!`);
    console.log(`📡 Listening on all network interfaces (0.0.0.0) on port ${port}\n`);
});
