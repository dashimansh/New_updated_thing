#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include "Config.h"
#include "Detector.h"
#include "Tracker.h"
#include "FrameStabilizer.h"
#include "ThreadedCamera.h"

class FPSCounter
{
public:
    void Tick()
    {
        auto N =
            std::chrono::steady_clock
            ::now();
        double E =
            std::chrono::duration<double>(
                N - Last).count();
        Last = N;
        FPS  = 1.0 / E;
    }
    double Get() const { return FPS; }

private:
    std::chrono::steady_clock::time_point
        Last =
        std::chrono::steady_clock::now();
    double FPS = 0.0;
};

void DrawHUD(
    cv::Mat& Frame,
    double FPS,
    ETrackState State,
    int Lost)
{
    cv::rectangle(
        Frame,
        cv::Rect(0, 0, Frame.cols, 75),
        cv::Scalar(0, 0, 0), -1);

    cv::putText(Frame,
        "FPS: " + std::to_string((int)FPS),
        cv::Point(10, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.7, cv::Scalar(0, 255, 255), 2);

    std::string S;
    cv::Scalar  SC;

    switch (State)
    {
    case ETrackState::Idle:
        S  = "IDLE";
        SC = cv::Scalar(200, 200, 200);
        break;
    case ETrackState::Tracking:
        S  = "TRACKING";
        SC = cv::Scalar(0, 255, 0);
        break;
    case ETrackState::Occluded:
        S  = "OCCLUDED [" +
            std::to_string(Lost) + "/" +
            std::to_string(MAX_LOST_FRAMES)
            + "]";
        SC = cv::Scalar(0, 165, 255);
        break;
    case ETrackState::Lost:
        S  = "LOST - press D";
        SC = cv::Scalar(0, 0, 255);
        break;
    }

    cv::putText(Frame,
        "STATE: " + S,
        cv::Point(160, 24),
        cv::FONT_HERSHEY_SIMPLEX,
        0.65, SC, 2);

    cv::putText(Frame,
        "S=Select D=Detect R=Reset "
        "T=Stabilize 1=CamShift "
        "2=Template Q=Quit",
        cv::Point(10, 55),
        cv::FONT_HERSHEY_SIMPLEX,
        0.45,
        cv::Scalar(180, 180, 180), 1);
}

int main()
{
    std::cout
        << "Object Tracker\n"
        << "==============\n"
        << "S = Select manually\n"
        << "D = Auto detect red\n"
        << "R = Reset\n"
        << "T = Stabilize toggle\n"
        << "1 = CamShift tracker\n"
        << "2 = Template tracker\n"
        << "Q = Quit\n\n";

    // ================================================
    // CHANGE YOUR OFFICE IP CAMERA URL HERE
    // Just replace the IP, username and password below
    // ================================================
    const std::string STREAM_URL =
        "rtsp://admin:admin123@192.168.1.64:554/stream";
    //         ^^^^^  ^^^^^^^^  ^^^^^^^^^^^^ ^^^
    //       username password     IP        port
    // ================================================

    ThreadedCamera  Camera;
    Detector        Det;
    ObjectTracker   Tracker;
    FrameStabilizer Stab;
    FPSCounter      FPS;

    std::cout
        << "Connecting to: "
        << STREAM_URL << "\n";

    if (!Camera.OpenFile(STREAM_URL))
    {
        std::cerr
            << "Failed to open camera: "
            << STREAM_URL << "\n"
            << "Check IP, username, password\n";
        return -1;
    }

    std::cout << "Camera opened!\n";

    cv::Mat Frame;
    int  FrameCount = 0;
    bool bStabilize = false;

    cv::namedWindow(
        WINDOW_NAME,
        cv::WINDOW_NORMAL);
    cv::resizeWindow(
        WINDOW_NAME, 960, 540);

    while (true)
    {
        if (!Camera.GetLatestFrame(Frame))
            continue;

        FrameCount++;
        FPS.Tick();

        cv::Mat PF = Frame.clone();

        if (bStabilize)
            PF = Stab.Stabilize(Frame);

        if (FrameCount %
            DETECTION_INTERVAL == 0
            && Tracker.IsTracking())
        {
            DetectionResult DR =
                Det.Detect(PF);
            if (DR.bFound)
                Det.AdaptColorRange(
                    PF, DR.BoundingBox);
        }

        if (Tracker.IsLost() ||
            (Tracker.IsOccluded() &&
                Tracker.GetLostCount() >
                MAX_LOST_FRAMES / 2))
        {
            DetectionResult DR =
                Det.Detect(PF);
            if (DR.bFound)
            {
                cv::Rect  Pred =
                    Tracker.GetPredictedBox();
                cv::Point PC(
                    Pred.x + Pred.width  / 2,
                    Pred.y + Pred.height / 2);
                float D = (float)cv::norm(
                    PC - DR.Center);
                if (D < MAX_JUMP_DISTANCE
                    || Tracker.IsLost())
                {
                    Tracker.Reinit(
                        PF, DR.BoundingBox);
                    std::cout
                        << "Recovered!\n";
                }
            }
        }

        if (Tracker.GetState() !=
            ETrackState::Idle)
            Tracker.Update(PF);

        Tracker.Draw(PF);
        DrawHUD(PF, FPS.Get(),
            Tracker.GetState(),
            Tracker.GetLostCount());

        cv::imshow(WINDOW_NAME, PF);

        char Key = (char)cv::waitKey(1);

        if (Key == 's' || Key == 'S')
        {
            cv::Rect ROI =
                Det.SelectROI(PF);
            if (ROI.width > 0 &&
                ROI.height > 0)
                Tracker.Init(PF, ROI,
                    ETrackerType::CAMSHIFT);
        }
        else if (Key == 'd' || Key == 'D')
        {
            DetectionResult DR =
                Det.Detect(PF);
            if (DR.bFound)
                Tracker.Init(PF,
                    DR.BoundingBox,
                    ETrackerType::CAMSHIFT);
            else
                std::cout << "Not found\n";
        }
        else if (Key == 'r' || Key == 'R')
        {
            Tracker.Reset();
            Stab.Reset();
            std::cout << "Reset\n";
        }
        else if (Key == 't' || Key == 'T')
        {
            bStabilize = !bStabilize;
            std::cout << "Stabilize: "
                << (bStabilize ?
                    "ON" : "OFF") << "\n";
        }
        else if (Key == '1')
        {
            if (Tracker.GetState() !=
                ETrackState::Idle)
                Tracker.Init(PF,
                    Tracker.GetCurrentBox(),
                    ETrackerType::CAMSHIFT);
        }
        else if (Key == '2')
        {
            if (Tracker.GetState() !=
                ETrackState::Idle)
                Tracker.Init(PF,
                    Tracker.GetCurrentBox(),
                    ETrackerType::TEMPLATE);
        }
        else if (Key == 'q' ||
            Key == 'Q' || Key == 27)
            break;
    }

    Camera.Stop();
    cv::destroyAllWindows();
    return 0;
}
