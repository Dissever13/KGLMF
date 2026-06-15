"""Minimal test to diagnose Illustrator drawing issues."""
import os, sys, pythoncom, win32com.client

pythoncom.CoInitialize()
app = win32com.client.Dispatch("Illustrator.Application")

doc = app.Documents.Add()
W, H = 600, 400

# ---- Test 1: Set artboard ----
js_setup = f"""
var doc = app.activeDocument;
doc.artboards[0].artboardRect = [0, {H}, {W}, 0];
"""
app.DoJavaScript(js_setup)
print("Test 1: Artboard set")

# ---- Test 2: Simple rectangle ----
js_rect = f"""
var doc = app.activeDocument;
var shape = doc.pathItems.rectangle({H - 50}, 50, 200, 100);
var fc = new RGBColor(); fc.red = 66; fc.green = 165; fc.blue = 245;
shape.fillColor = fc;
shape.filled = true;
shape.stroked = false;
"""
try:
    app.DoJavaScript(js_rect)
    print("Test 2: Rectangle drawn OK")
except Exception as e:
    print(f"Test 2 FAILED: {e}")

# ---- Test 3: Text via textFrames.add() ----
js_text1 = f"""
var doc = app.activeDocument;
var tf = doc.textFrames.add();
tf.contents = 'Hello World (add)';
tf.position = [100, {H - 100}];
tf.textRange.characterAttributes.size = 24;
"""
try:
    app.DoJavaScript(js_text1)
    print("Test 3: Text frame (add) OK")
except Exception as e:
    print(f"Test 3 FAILED: {e}")

# ---- Test 4: Text via pointText ----
js_text2 = f"""
var doc = app.activeDocument;
var pt = doc.textFrames.pointText([300, {H - 150}]);
pt.contents = 'Hello World (pointText)';
pt.textRange.characterAttributes.size = 24;
"""
try:
    app.DoJavaScript(js_text2)
    print("Test 4: pointText OK")
except Exception as e:
    print(f"Test 4 FAILED: {e}")

# ---- Test 5: List available fonts ----
js_fonts = """
var doc = app.activeDocument;
var fonts = app.textFonts;
var result = '';
for (var i = 0; i < Math.min(10, fonts.length); i++) {
    result += fonts[i].name + ' | ';
}
result;
"""
try:
    result = app.DoJavaScript(js_fonts)
    print(f"Test 5: First 10 fonts: {result}")
except Exception as e:
    print(f"Test 5 FAILED: {e}")

# ---- Test 6: Try with font name ----
js_text3 = f"""
var doc = app.activeDocument;
var tf = doc.textFrames.add();
tf.contents = 'Hello with Arial';
tf.position = [100, {H - 200}];
tf.textRange.characterAttributes.size = 20;
try {{
    tf.textRange.characterAttributes.textFont = textFonts.getByName('Arial');
}} catch(e) {{
    // ignore font error
}}
"""
try:
    app.DoJavaScript(js_text3)
    print("Test 6: Text with font OK")
except Exception as e:
    print(f"Test 6 FAILED: {e}")

# ---- Test 7: Rounded rectangle ----
js_rrect = f"""
var doc = app.activeDocument;
var shape = doc.pathItems.roundedRectangle({H - 80}, 350, 180, 60, 8, 8);
var fc = new RGBColor(); fc.red = 255; fc.green = 128; fc.blue = 0;
shape.fillColor = fc;
shape.filled = true;
"""
try:
    app.DoJavaScript(js_rrect)
    print("Test 7: Rounded rect OK")
except Exception as e:
    print(f"Test 7 FAILED: {e}")

# ---- Test 8: Arrow line ----
js_arrow = f"""
var doc = app.activeDocument;
var line = doc.pathItems.add();
line.setEntirePath([[50, {H - 300}], [300, {H - 300}]]);
var sc = new RGBColor(); sc.red = 0; sc.green = 0; sc.blue = 0;
line.strokeColor = sc;
line.strokeWidth = 3;
line.filled = false;
line.stroked = true;
"""
try:
    app.DoJavaScript(js_arrow)
    print("Test 8: Arrow line OK")
except Exception as e:
    print(f"Test 8 FAILED: {e}")

# Export test
test_pdf = r"D:\KGLM\KGLMF\paper\figures\_test_debug.pdf"
abs_path = test_pdf.replace("\\", "/")
try:
    doc.Export(abs_path, 3, {})
    size = os.path.getsize(test_pdf)
    print(f"\nTest PDF exported: {test_pdf} ({size} bytes)")
except Exception as e:
    print(f"Export failed: {e}")

print("\n=== DEBUG COMPLETE ===")
print("Please open _test_debug.pdf to see what rendered.")
