# Manual Court Calibration Guide

## 🎯 What You'll See

When you run the manual calibration tool, here's what happens:

---

## Option 1: Simple 4-Corner Calibration

### Command:
```bash
python manual_court_calibration.py --video your_video.mov
```

### What You Do:
Click **4 corners** of the court in this order:

```
         1 ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━● 2
           ┃         FAR COURT           ┃
           ┃                             ┃
           ┃         (Players)           ┃
           ┃                             ┃
           ┃        NEAR COURT           ┃
         4 ●━━━━━━━━━━━━━━━━━━━━━━━━━━━━● 3

Click order:
  1. Top-Left (far baseline left)
  2. Top-Right (far baseline right)  
  3. Bottom-Right (near baseline right)
  4. Bottom-Left (near baseline left)
```

### What Happens:
- ✅ Window auto-resizes to fit your screen
- ✅ Green circles show where you clicked
- ✅ Lines connect the points as you click
- ✅ Coordinates saved in original video size
- ✅ Press **'s'** to save when done

### Output:
`court_calibration.json` with 4 corner points

---

## Option 2: Complete Line Definition (More Precise!)

### Command:
```bash
python manual_court_lines_full.py --video your_video.mov
```

### What You Do:
Click **2 points per line** (20 points total) for these lines:

```
                    SINGLES
                    LINE
         ●━━━━━━━━━━┃━━━━━━━━━━━●
         ┃          ┃            ┃  ← Far Baseline (Line 1)
         ┃    ●━━━━━●━━━━━●      ┃  ← Top Service Line (Line 5)
         ┃    ┃            ┃     ┃
    Left ┃    ┃    NET     ┃     ┃ Right
 Sideline┃    ┃   (Line 8) ┃     ┃ Sideline
 (Line 3)┃    ┃            ┃     ┃ (Line 4)
         ┃    ●━━━━━●━━━━━●      ┃  ← Bottom Service Line (Line 6)
         ┃          ┃            ┃
         ┃          ┃ Center     ┃
         ●━━━━━━━━━━┃━━━━━━━━━━━●  ← Near Baseline (Line 2)
                    ↑
                Center Service
                Line (Line 7)
```

### Lines You'll Define:
1. **Far Baseline** (top) - 2 points
2. **Near Baseline** (bottom) - 2 points
3. **Left Sideline** - 2 points
4. **Right Sideline** - 2 points
5. **Top Service Line** - 2 points
6. **Bottom Service Line** - 2 points
7. **Center Service Line** - 2 points
8. **Net Line** - 2 points
9. **Left Singles Line** (optional) - 2 points
10. **Right Singles Line** (optional) - 2 points

### Controls:
- **Click** = Place point
- **'s'** = Save (after all lines done)
- **'r'** = Reset and start over
- **'n'** = Skip (optional lines only)
- **'q'** = Quit without saving

### What Happens:
- ✅ Prompts you for each line one at a time
- ✅ Shows progress (e.g., "Line 3/10")
- ✅ Green circles mark your points
- ✅ Green lines connect the 2 points for each line
- ✅ Can skip singles lines if not needed

### Output:
`court_lines_manual.json` with all line definitions

---

## 🎨 Visual Example

### When You Click:

```
Before clicking:          After 1 click:           After 2 clicks (line complete):
┌───────────────┐        ┌───────────────┐        ┌───────────────┐
│               │        │ ●P1           │        │ ●P1━━━━━━━●P2 │
│               │   →    │               │   →    │               │
│               │        │               │        │               │
└───────────────┘        └───────────────┘        └───────────────┘
                                                   ✅ Line Complete!
```

### On Screen You'll See:
```
📍 Line 1/10: Far Baseline (2 points: left to right)
   Click 2 points...
   Point 1: (150, 100)
   Point 2: (950, 105)
✅ Far Baseline - COMPLETE

📍 Line 2/10: Near Baseline (2 points: left to right)
   Click 2 points...
```

---

## 📊 Comparison

| Feature | 4-Corner (Simple) | Full Lines (Precise) |
|---------|------------------|---------------------|
| **Points to click** | 4 | 20 (10 lines × 2 points) |
| **Time** | ~10 seconds | ~2 minutes |
| **Accuracy** | Good (calculated) | Excellent (exact) |
| **Service lines** | Calculated (~27%) | Exact position |
| **Net line** | Calculated (50%) | Exact position |
| **Best for** | Quick analysis | Professional accuracy |

---

## 🚀 Try It Now!

### Quick Test (4 corners):
```bash
cd tennis-ai-main
python manual_court_calibration.py --video copy_9DE8D780-1898-4AA1-839E-7FAC52A6D63B.mov
```

### Full Definition (all lines):
```bash
cd tennis-ai-main
python manual_court_lines_full.py --video copy_9DE8D780-1898-4AA1-839E-7FAC52A6D63B.mov
```

---

## ✅ What Happens After Calibration

Once you save, use it in your analysis:

```bash
# With 4-corner calibration:
python main_pose.py --video your_video.mov \
                    --court-calibration court_calibration.json \
                    --output analysis.mp4

# With full line definition:
python main_pose.py --video your_video.mov \
                    --court-lines court_lines_manual.json \
                    --output analysis.mp4
```

Both methods give you:
- ✅ Perfect court line overlays
- ✅ Accurate speed measurements
- ✅ Rally analysis
- ✅ In/out detection

**The full line method is recommended for best accuracy!** 🎾

