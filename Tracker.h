#pragma once
#include <opencv2/opencv.hpp>
#include "KalmanTracker.h"

enum class ETrackerType
{
    CAMSHIFT,
    TEMPLATE
};

enum class ETrackState
{
    Idle,
    Tracking,
    Occluded,
    Lost
};

class ObjectTracker
{
public:
    ObjectTracker();

    bool Init(
        const cv::Mat& Frame,
        const cv::Rect& BBox,
        ETrackerType Type =
            ETrackerType::CAMSHIFT);

    cv::Rect Update(
        const cv::Mat& Frame);

    bool Reinit(
        const cv::Mat& Frame,
        const cv::Rect& NewBBox);

    void Reset();
    void Draw(cv::Mat& Frame);

    ETrackState GetState() const
    {
        return State;
    }
    cv::Rect GetCurrentBox() const
    {
        return CurrentBox;
    }
    cv::Rect GetPredictedBox() const
    {
        return KFilter.GetPredictedBox();
    }
    int GetLostCount() const
    {
        return LostFrameCount;
    }
    bool IsTracking() const
    {
        return State ==
            ETrackState::Tracking;
    }
    bool IsOccluded() const
    {
        return State ==
            ETrackState::Occluded;
    }
    bool IsLost() const
    {
        return State ==
            ETrackState::Lost;
    }

private:
    KalmanTracker KFilter;
    ETrackState   State;
    ETrackerType  TrackerType;
    cv::Rect      CurrentBox;
    int           LostFrameCount = 0;

    cv::Mat  RoiHist;
    cv::Rect TrackWindow;
    cv::Mat  ObjTemplate;

    bool UpdateCamShift(
        const cv::Mat& Frame,
        cv::Rect& OutBox);

    bool UpdateTemplate(
        const cv::Mat& Frame,
        cv::Rect& OutBox);
};
