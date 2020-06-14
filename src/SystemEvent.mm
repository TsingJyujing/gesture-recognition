#import <Foundation/Foundation.h>
#include "SystemEvent.h"

/**
 * Open Mission Control
 * TODO check is there any mission control in path before call it
 */
void MacSystemCall::openMissionControl() {
    NSLog(@"Open Mission Control");
    system("open /System/Applications/Mission\\ Control.app");
}

void MacSystemCall::pressSingleKey(int keyCode) {
    @autoreleasepool {
        CGEventSourceRef src = CGEventSourceCreate(kCGEventSourceStateHIDSystemState);
        CGEventRef keyDown = CGEventCreateKeyboardEvent(src, (CGKeyCode) keyCode, true);
        CGEventRef keyUp = CGEventCreateKeyboardEvent(src, (CGKeyCode) keyCode, false);
        CGEventTapLocation loc = kCGHIDEventTap; // kCGSessionEventTap also works
        CGEventPost(loc, keyDown);
        usleep(1000);
        CGEventPost(loc, keyUp);
        CFRelease(keyDown);
        CFRelease(keyUp);
        CFRelease(src);
    }
}