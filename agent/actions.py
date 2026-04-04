import pygame
import os
import time

def play_alert_sound(**kwargs):
    # Flexible argument handling
    level = kwargs.get("level") or kwargs.get("alert_type") or kwargs.get("type") or "high"
    
    print(f"[ACTION] Playing {level} alert sound")
    
    # Initialize mixer if not already done
    if not pygame.mixer.get_init():
        pygame.mixer.init()
        
    # Generate a beep sound programmatically to avoid missing file issues
    # or play a file if it exists. For now, let's assume we might not have the file
    # and fail gracefully or just print.
    # Ideally, we should have an 'alert.wav' in assets.
    
    # For this MVP, let's try to load a file, if not, print warning.
    sound_path = os.path.join("frontend", "assets", "alert.mp3") # Hypothetical path
    
    if os.path.exists(sound_path):
        try:
            pygame.mixer.music.load(sound_path)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(0.1)
        except Exception as e:
            print(f"⚠️ Could not play sound: {e}")
    else:
        print(f"⚠️ Alert sound file not found at {sound_path}")
        # Fallback: Just print loud visual alert
        print("BEEP! BEEP! BEEP!")

    return {"status": "played", "level": level}

# Register actions
ACTIONS = {
    "play_alert_sound": play_alert_sound
}