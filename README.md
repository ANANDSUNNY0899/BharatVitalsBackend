# Bharat Vitals Backend

This is the backend server for the **Bharat Vitals** application. It is built using Node.js and Express, integrating with the **Google Gemini API** (`gemini-2.5-flash`) to analyze images of Indian Thalis and identify specific food items for metabolic health analysis.

---

## Features

- **Food Recognition API**: Detects key items from an Indian Thali (Rice, Roti, Dal, Paneer, Curd, Sabzi, Potato).
- **Mobile-Friendly Output**: Returns a strict, lightweight JSON array that is clean and ready for consumption by mobile applications (e.g., Android apps).
- **Network Accessibility**: The server listens on all network interfaces (`0.0.0.0`), allowing devices on the same Wi-Fi network (like physical Android phones) to connect directly.

---

## Prerequisites

Ensure you have the following installed on your machine:
- [Node.js](https://nodejs.org/) (v18.x or higher recommended)
- [npm](https://www.npmjs.com/) (Node Package Manager)

---

## Setup & Installation

1. **Install Dependencies**:
   Navigate to the project root and install the required npm packages:
   ```bash
   npm install
   ```

2. **Environment Variables**:
   Create a `.env` file in the root directory (one has been automatically created for you) and add your Gemini API Key:
   ```env
   GEMINI_API_KEY=your_gemini_api_key_here
   ```

---

## Running the Server

Start the Express backend server by running:
```bash
node server.js
```

Upon success, you should see:
```text
🚀 Bharat Vitals Cloud Vision Server is awake!
📡 Listening on all network interfaces (0.0.0.0) on port 3000
```

---

## API Reference

### Vision Scan Endpoint

Analyzes a base64 encoded image of a meal (specifically an Indian Thali) and returns a list of identified food items.

- **URL**: `/api/vision-scan`
- **Method**: `POST`
- **Headers**: `Content-Type: application/json`
- **Request Body**:
  ```json
  {
    "imageBase64": "data:image/jpeg;base64,/9j/4AAQSkZJRg..."
  }
  ```
- **Response** (Success - `200 OK`):
  ```json
  [
    { "name": "Rice" },
    { "name": "Dal" },
    { "name": "Roti" }
  ]
  ```

*Note: If the analysis fails or no matching foods are found, it returns an empty array `[]` to prevent mobile application crashes.*

---

## License

This project is licensed under the ISC License.
