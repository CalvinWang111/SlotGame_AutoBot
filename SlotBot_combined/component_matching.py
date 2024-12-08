class ComponentMatcher:
    @staticmethod
    def match_components(masks, predictions):
        """匹配辨識結果與分割元件位置"""
        matched_components = []
        for mask, prediction in zip(masks, predictions):
            matched_components.append({
                "label": prediction.item(),
                "position": mask["bbox"]
            })
        return matched_components
