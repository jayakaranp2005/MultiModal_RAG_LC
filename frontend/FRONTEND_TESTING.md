# Frontend Testing Checklist

Use this checklist in your browser DevTools Network tab while running `npm run dev` in `frontend/`.

## 1) Initial load calls
- Open the app at `http://localhost:8080`.
- Confirm `GET /health` is called on page load.
- Confirm `GET /indexed` is called on page load.

## 2) Indexed refresh
- Click the **Refresh** button in the **Indexed PDFs** section.
- Confirm another `GET /indexed` request is made.

## 3) Upload endpoint uses multipart/form-data
- Choose a `.pdf` file and click **Upload & Index**.
- Confirm request is `POST /upload_pdf?reindex=true|false`.
- In request headers, confirm `Content-Type` is `multipart/form-data; boundary=...`.
- Confirm request payload includes form-data key `file`.

## 4) Ask endpoint sends JSON
- Enter a question and click **Ask** (or press Ctrl+Enter).
- Confirm request is `POST /ask`.
- Confirm request header includes `Content-Type: application/json`.
- Confirm request body is JSON with `{ "question": "..." }`.
- Confirm response JSON includes `answer` and `image_count`.

## 5) Basic UX behavior
- Verify buttons disable during in-flight requests.
- Verify error messages appear for invalid file, empty question, or backend failures.
- Verify answer text preserves line breaks and does not render HTML.
