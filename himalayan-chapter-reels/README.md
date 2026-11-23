# The Chapters: Reel Engine for The Himalayan Chapter

*"This café in Tapovan doesn't have customers. It has chapters."*

This folder turns a phone clip of a guest into a finished, on-brand vertical reel (1080×1920) with:
cleaned voice, subtitles (translated to English when the guest speaks Hindi, Hebrew, etc.), a chapter
title, smooth dissolves, an original copyright-free soundtrack that ducks under speech, and the logo end card.

```
engine/reel.py         builds a reel from an episode file (JSON)
engine/music.py        original soundtrack generator (nostalgic lo-fi: e-piano, guitar, glockenspiel, bansuri, vinyl)
engine/menu_reel.py    menu showcase reel (page flips, highlighter, dish tags)
engine/event_reel.py   event promo (poster layout, lineup drop, night-sky booking card)
engine/event_scene_reel.py  event promo drawn entirely in code: moonrise -> vinyl moon, lasers, crowd
engine/alien_reel.py   event story reel: alien DJs intercept the signal, land on the rooftop, drop the beat
engine/music_edm.py    melodic-house event track: build, drop, breakdown
engine/music_himalayan.py  Himalayan-café score: tanpura, santoor, bansuri (Raag Bhupali), tabla (keherwa)
tools/prepare_clip.py  cleans a new clip's audio, transcribes it, writes a draft daily episode
episodes/              one JSON file per reel (ep01_launch.json, daily_ch05.json, ...)
assets/                logo + brand fonts (Cinzel, Cormorant Garamond; SIL Open Font License)
footage/               raw clips + footage/clean/*.wav (not committed: guest faces, large files)
out/                   rendered videos (not committed)
```

## Menu launch reel

`python engine/menu_reel.py episodes/menu_launch.json` → `out/THC_New_Menu_Launch.mp4`

The camera glides over the illustrated menu pages (`assets/menu/`), pages flip like a book, a gold
highlighter draws around featured dishes and a brand tag shows dish and price. To feature different
dishes, edit the `beats` in the episode file: `cam0`/`cam1` are [x, y, visible width] in page pixels
(page = 1055×1491), `hl` rectangles are [x0, y0, x1, y1] in page pixels, and `tag` is name/desc/price.
Beats with `"glide": true` move the camera on the same page instead of flipping. `"music": "himalayan"`
selects the raag score; `music_marks` says which beats bring in the flute, the tabla and the quieter break.
Keep each stop at 5 s or more so people can read the tag.

## Event promo reel

`python engine/event_reel.py episodes/event_fullmoon_2411.json` → `out/THC_Full_Moon_DJ_Night_24Nov.mp4`

For the next event, copy the episode file and change the title, artists, date/time lines, phone and
`segments` (which seconds of the footage to use, slowed to `dur`, and a `crop` [w, h, x, y] that keeps the
2.4:1 strip clear of any burned-in text). The music drops on the second segment, when the lineup appears.

`python engine/event_scene_reel.py episodes/event_fullmoon_2411_v2.json` builds the footage-free version
("The Moon Drops the Beat"): change the text fields, `drop`, `lineup_at`, `date_at` and `break` for a new event.

`python engine/alien_reel.py episodes/event_fullmoon_2411_aliens.json` builds the alien story version
("Signal from Tapovan"). `times` = [space, arrival, beam, drop, info, punchline, end]; the HUD lines,
story lines and end hook are all text fields in the episode file.

## Daily workflow (about 10 minutes per chapter)

1. **Film** (see the filming guide below) and copy the clip to `footage/`, named like `ch06_masala_chai.mp4`.
2. **Prepare:** `python tools/prepare_clip.py footage/ch06_masala_chai.mp4 --num 06 --title "The Masala Chai"`
   This prints the transcript with word timings and writes `episodes/daily_ch06.json`.
3. **Edit the episode file.** Pick the best seconds in `parts` (lead with the strongest line, which makes a
   better hook), tidy the subtitle text, and put a word or two in the 4th field of a subtitle to colour it gold.
4. **Render:** `python engine/reel.py episodes/daily_ch06.json` writes `out/THC_Daily_Chapter06.mp4` at −14 LUFS.

### Episode file reference

| shot `type` | fields |
|---|---|
| `hook` | `src`, `in`, `out`, `kicker`, `lines` (big italic lines), optional `pre` (ffmpeg filters, e.g. a slow zoom) |
| `card` | `dur`, `small`, `big`, `foot`: cream paper page that turns in |
| `chapter` | `num`, `title`, `src`, `voice` (clean wav), `parts` [[start,end],…] in source seconds (use several for jump cuts or to reorder), `subs` [[start,end,text,highlight],…] in source seconds, optional `pre` (e.g. `transpose=1,` for a sideways clip, `crop=…,` to hide something) |
| `end` | `dur`, `line1`, `line2`, `info` (list of lines), `place` |

Transitions are automatic: a page turn into cards and the end card, a 0.35 s dissolve between shots, and a
short dissolve between parts of the same chapter. Soft swish sounds follow the cuts.

## Filming guide for the team

- **Vertical, at eye level, 1–1.5 m away**, guest facing a window (soft light). Hold still for 15–20 s.
- Get the **dish in frame** with the guest. Food sells the reel.
- Ask one question and let them talk: *"What did this place remind you of?"* or *"What would you tell a
  friend about this place?"* Then catch their name and where they're from.
- Keep the café quiet for those 20 seconds (music down). A ₹1,500 clip-on mic makes a big difference.
- **Ask permission** to post, and tag them, so they share it, which is free reach.
- Keep other brands' menus or logos out of frame.
- Hand them the Chapter Book and film them writing one line; that makes a great opening shot.

## Weekly rhythm (one reel a day)

| Day | Format |
|---|---|
| Mon, Wed, Fri, Sun | **Chapter NN**: one guest, one story (this engine) |
| Tue | **Taste of Childhood**: a dish plus the memory it brings back |
| Thu | **Wings Over Tapovan**: the 5 PM sky over the rooftop |
| Sat | **Rooftop Sessions**: artist or event teaser beforehand, recap afterwards |

## Setup

```
pip install numpy scipy pillow fonttools faster-whisper
# ffmpeg must be on PATH. For the best voice cleanup also install the DeepFilterNet binary
# (https://github.com/Rikorose/DeepFilterNet/releases) as `deep-filter` on PATH.
```
The first transcription downloads the Whisper "medium" model (~1.5 GB) from Hugging Face.
