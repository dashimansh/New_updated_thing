#include "Tracker.h"
#include "Config.h"
#include <iostream>

ObjectTracker::ObjectTracker()
{
    State         = ETrackState::Idle;
    LostFrameCount = 0;
}

bool ObjectTracker::Init(
    const cv::Mat& Frame,
    const cv::Rect& BBox,
    ETrackerType Type)
{
    TrackerType = Type;

    cv::Rect SafeBox = BBox &
        cv::Rect(0, 0,
            Frame.cols, Frame.rows);
    if (SafeBox.area() <= 0)
        return false;

    cv::Mat HSV;
    cv::cvtColor(Frame, HSV,
        cv::COLOR_BGR2HSV);

    cv::Mat ROI = HSV(SafeBox);
    int   HistSize = 16;
    float Range[]  = { 0, 180 };
    const float* Ranges = Range;
    int Channels = 0;

    cv::calcHist(&ROI, 1, &Channels,
        cv::Mat(), RoiHist,
        1, &HistSize, &Ranges);
    cv::normalize(RoiHist, RoiHist,
        0, 255, cv::NORM_MINMAX);

    ObjTemplate = Frame(SafeBox).clone();
    TrackWindow = SafeBox;
    CurrentBox  = SafeBox;
    State       = ETrackState::Tracking;
    LostFrameCount = 0;

    KFilter.Init(SafeBox);

    std::cout << "Tracker initialized\n";
    return true;
}

bool ObjectTracker::UpdateCamShift(
    const cv::Mat& Frame,
    cv::Rect& OutBox)
{
    if (RoiHist.empty()) return false;

    cv::Mat HSV, BackProj;
    cv::cvtColor(Frame, HSV,
        cv::COLOR_BGR2HSV);

    float Range[] = { 0, 180 };
    const float* Ranges = Range;
    int Channels = 0;

    cv::calcBackProject(&HSV, 1,
        &Channels, RoiHist,
        BackProj, &Ranges);

    cv::TermCriteria TC(
        cv::TermCriteria::EPS |
        cv::TermCriteria::COUNT,
        10, 1);

    cv::Rect SearchWin = TrackWindow;
    SearchWin.x      -= 20;
    SearchWin.y      -= 20;
    SearchWin.width  += 40;
    SearchWin.height += 40;
    SearchWin &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);

    if (SearchWin.area() <= 0)
        return false;

    cv::Mat SearchBP = BackProj(SearchWin);
    cv::Rect LocalWin(
        0, 0,
        SearchWin.width,
        SearchWin.height);

    cv::RotatedRect RR =
        cv::CamShift(SearchBP,
            LocalWin, TC);

    OutBox    = RR.boundingRect();
    OutBox.x += SearchWin.x;
    OutBox.y += SearchWin.y;
    OutBox   &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);

    if (OutBox.area() < 100)
        return false;

    TrackWindow = OutBox;

    cv::Mat Region = BackProj(
        OutBox & cv::Rect(0, 0,
            Frame.cols, Frame.rows));
    cv::Scalar Mean = cv::mean(Region);
    return Mean[0] > 30.0;
}

bool ObjectTracker::UpdateTemplate(
    const cv::Mat& Frame,
    cv::Rect& OutBox)
{
    if (ObjTemplate.empty())
        return false;

    int Pad = 60;
    cv::Rect SearchArea(
        CurrentBox.x - Pad,
        CurrentBox.y - Pad,
        CurrentBox.width  + Pad * 2,
        CurrentBox.height + Pad * 2);
    SearchArea &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);

    if (SearchArea.width  < ObjTemplate.cols ||
        SearchArea.height < ObjTemplate.rows)
        return false;

    cv::Mat SearchROI = Frame(SearchArea);
    cv::Mat Result;
    cv::matchTemplate(SearchROI,
        ObjTemplate, Result,
        cv::TM_CCOEFF_NORMED);

    double MinVal, MaxVal;
    cv::Point MinLoc, MaxLoc;
    cv::minMaxLoc(Result,
        &MinVal, &MaxVal,
        &MinLoc, &MaxLoc);

    if (MaxVal < 0.5) return false;

    OutBox = cv::Rect(
        SearchArea.x + MaxLoc.x,
        SearchArea.y + MaxLoc.y,
        ObjTemplate.cols,
        ObjTemplate.rows);
    OutBox &= cv::Rect(0, 0,
        Frame.cols, Frame.rows);

    return true;
}

cv::Rect ObjectTracker::Update(
    const cv::Mat& Frame)
{
    if (State == ETrackState::Idle)
        return cv::Rect();

    cv::Rect TrackerBox;
    bool bOK = false;

    bOK = UpdateCamShift(Frame, TrackerBox);

    if (!bOK)
        bOK = UpdateTemplate(
            Frame, TrackerBox);

    if (bOK)
    {
        cv::Point LastC(
            CurrentBox.x +
            CurrentBox.width  / 2,
            CurrentBox.y +
            CurrentBox.height / 2);
        cv::Point NewC(
            TrackerBox.x +
            TrackerBox.width  / 2,
            TrackerBox.y +
            TrackerBox.height / 2);
        float Dist =
            (float)cv::norm(LastC - NewC);
        if (Dist > MAX_JUMP_DISTANCE)
            bOK = false;
    }

    if (bOK)
    {
        CurrentBox =
            KFilter.Update(TrackerBox);
        State          = ETrackState::Tracking;
        LostFrameCount = 0;
    }
    else
    {
        LostFrameCount++;
        if (LostFrameCount <= MAX_LOST_FRAMES)
        {
            CurrentBox = KFilter.Predict();
            State      = ETrackState::Occluded;
            std::cout << "Occluded ["
                << LostFrameCount << "]\n";
        }
        else
        {
            State = ETrackState::Lost;
            std::cout << "Object LOST\n";
        }
    }
    return CurrentBox;
}

bool ObjectTracker::Reinit(
    const cv::Mat& Frame,
    const cv::Rect& NewBBox)
{
    return Init(Frame, NewBBox, TrackerType);
}

void ObjectTracker::Reset()
{
    KFilter.Reset();
    RoiHist.release();
    ObjTemplate.release();
    State          = ETrackState::Idle;
    LostFrameCount = 0;
    CurrentBox     = cv::Rect();
}

void ObjectTracker::Draw(cv::Mat& Frame)
{
    if (State == ETrackState::Idle)
        return;

    cv::Scalar  Color;
    std::string Text;

    switch (State)
    {
    case ETrackState::Tracking:
        Color = cv::Scalar(0, 255, 0);
        Text  = "TRACKING";
        break;
    case ETrackState::Occluded:
        Color = cv::Scalar(0, 165, 255);
        Text  = "OCCLUDED";
        break;
    case ETrackState::Lost:
        Color = cv::Scalar(0, 0, 255);
        Text  = "LOST";
        break;
    default: return;
    }

    cv::rectangle(Frame,
        CurrentBox, Color, 2);

    if (SHOW_KALMAN_PRED &&
        State == ETrackState::Occluded)
    {
        cv::Rect P =
            KFilter.GetPredictedBox();
        cv::rectangle(Frame, P,
            cv::Scalar(255, 255, 0), 1);
        cv::putText(Frame, "KALMAN",
            cv::Point(P.x, P.y - 5),
            cv::FONT_HERSHEY_SIMPLEX,
            0.4,
            cv::Scalar(255, 255, 0), 1);
    }

    cv::Point C(
        CurrentBox.x + CurrentBox.width  / 2,
        CurrentBox.y + CurrentBox.height / 2);

    cv::line(Frame,
        cv::Point(C.x - 15, C.y),
        cv::Point(C.x + 15, C.y),
        Color, 2);
    cv::line(Frame,
        cv::Point(C.x, C.y - 15),
        cv::Point(C.x, C.y + 15),
        Color, 2);

    cv::putText(Frame, Text,
        cv::Point(CurrentBox.x,
            CurrentBox.y - 10),
        cv::FONT_HERSHEY_SIMPLEX,
        0.6, Color, 2);
}
