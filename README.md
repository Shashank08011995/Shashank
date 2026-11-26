# Experience Rishikesh

Website for **experiencerishikesh.com**: a complete one-day Rishikesh experience with a scooty, local guide, a special hidden place, food, sunset by the Ganga and live music in the evening. Everything is included and there are no commitments.

It's a static site (plain HTML and CSS, no build step).

## Before going live

1. **WhatsApp number**: in `index.html`, set `WHATSAPP_NUMBER` (digits only, with country code, e.g. `919876543210`).
2. **Price**: replace `₹____` in the booking section.
3. **Photos**: add these to an `images/` folder. Until you do, the site shows coloured gradients.
   - `hero.jpg`: wide landscape shot for the top of the page (ideally 2000px wide)
   - `gallery-1.jpg` … `gallery-6.jpg`: road, special place, sunset, live music, food, group
4. **Links**: update the Instagram URL and email in the footer.
5. Check the itinerary times and FAQ answers against how you actually run the day.

## Preview locally

Open `index.html` in a browser, or run:

```
python3 -m http.server 8000
```

## Deploy

Any static host works: Netlify, Vercel, GitHub Pages or Cloudflare Pages. Point the experiencerishikesh.com domain at it.
