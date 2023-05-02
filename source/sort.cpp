/********************************************************************************
** @auth£º taify
** @date£º 2023/04/21
** @Ver :  V1.0.0
** @desc£º sortÍ·ÎÄ¼þ
*********************************************************************************/


#include "utils.h"
#include "kalmanboxtracker.h"
#include "sort.h"


Sort::Sort(int max_age = 1, int min_hits = 3, float iou_threshold = 0.3)
{
	m_max_age = max_age;
	m_min_hits = min_hits;
	m_iou_threshold = iou_threshold;
	m_trackers = {};
	m_frame_count = 0;
}


std::vector<std::vector<float>> Sort::update(std::vector<cv::Rect> dets)
{
	m_frame_count++;
	std::vector<std::vector<int>> trks(m_trackers.size(), std::vector<int>(5, 0));
	std::vector<int> to_del;
	std::vector<std::vector<float>> ret;

	for (size_t i = 0; i < trks.size(); i++)
	{
		std::vector<float> pos = m_trackers[i].predict();
		std::vector<int> trk{ (int)pos[0],  (int)pos[1], (int)pos[2], (int)pos[3], 0 };
		trks[i] = trk;
	}

	for (int i = to_del.size() - 1; i >= 0; i--)
		m_trackers.erase(m_trackers.begin() + i);

	auto [matched, unmatched_dets, unmatched_trks] = associate_detections_to_tracks(dets, trks, m_iou_threshold);

	for (size_t i = 0; i < matched.size(); i++)
	{
		int id = matched[i].first;
		std::vector<int> vec{ dets[id].x, dets[id].y, dets[id].x + dets[id].width, dets[id].y + dets[id].height };
		m_trackers[matched[i].second].update(vec);
	}

	for (auto i : unmatched_dets)
	{
		std::vector<int> vec{ dets[i].x, dets[i].y, dets[i].x + dets[i].width, dets[i].y + dets[i].height };
		KalmanBoxTracker trk(vec);
		m_trackers.push_back(trk);
	}
	int n = m_trackers.size();

	for (int i = m_trackers.size() - 1; i >= 0; i--)
	{
		auto trk = m_trackers[i];
		//std::cout << KalmanBoxTracker::count << std::endl;
		std::vector<float> d = trk.get_state();
		if ((trk.m_time_since_update < 1) && (trk.m_hit_streak >= m_min_hits || m_frame_count <= m_min_hits))
		{
			std::vector<float> tmp = d;
			tmp.push_back(trk.m_id + 1);
			ret.push_back(tmp);
		}
		n--;
		if (trk.m_time_since_update > m_max_age)
			m_trackers.erase(m_trackers.begin() + n);
	}

	if (ret.size() > 0)
		return ret;

	return {};
}